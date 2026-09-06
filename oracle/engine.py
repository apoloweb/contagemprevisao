"""OracleEngine: roda o oraculo numa thread e expoe frame anotado + estado.

Diferente do app.py (CLI/gravacao), aqui o loop roda em background e:
  - guarda o ultimo frame anotado como JPEG (para MJPEG no navegador)
  - guarda o estado ao vivo (rodada, tempo, contagem) num dict protegido por lock
  - chama on_round_end(result) quando uma rodada fecha (para liquidar o mercado)
  - aceita mudancas ao vivo: linha de contagem, classes e limiar (threshold)
"""

from __future__ import annotations

import threading
import time

import cv2

from .counting import LineCounter
from .overlay import draw_boxes, draw_hud, draw_line
from .round import RoundManager, utc_now_iso
from .source import resolve_source

COCO = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
VEHICLES = ["car", "motorcycle", "bus", "truck"]


def _open_capture(target):
    cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG) if isinstance(target, str) else cv2.VideoCapture(target)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass
    return cap


def line_from_frac(frac, w, h):
    x1, y1, x2, y2 = frac
    return ((x1 * w, y1 * h), (x2 * w, y2 * h))


class OracleEngine(threading.Thread):
    def __init__(self, cfg: dict, on_round_end=None):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.on_round_end = on_round_end
        self._lock = threading.Lock()
        self._jpeg = None
        self._state = {"ready": False}
        self._stop = threading.Event()

        self._classes = list(cfg.get("classes", VEHICLES))
        self._threshold = int(cfg.get("threshold", 5))
        self._question = ""
        self._line_frac = tuple(cfg.get("line_frac", (0.5, 0.05, 0.5, 0.65)))
        self._pending_line = None
        self._pending_classes = None
        self._pending_threshold = None
        self._jpeg_quality = int(cfg.get("jpeg_quality", 70))

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def set_line_frac(self, x1, y1, x2, y2):
        self._pending_line = (float(x1), float(y1), float(x2), float(y2))

    def set_classes(self, names):
        self._pending_classes = [n for n in names if n in COCO]

    def set_threshold(self, x):
        self._pending_threshold = int(x)

    def set_question(self, q):
        self._question = q

    def stop(self):
        self._stop.set()

    def run(self):
        from ultralytics import YOLO

        target = resolve_source(str(self.cfg["source"]), int(self.cfg.get("max_height", 720)))
        model = YOLO(self.cfg.get("model", "yolo11s.pt"))
        device = self.cfg.get("device", 0)
        conf = float(self.cfg.get("conf", 0.3))
        round_seconds = float(self.cfg.get("round_seconds", 60))

        cap = _open_capture(target)
        ok, frame = cap.read()
        while (not ok or frame is None) and not self._stop.is_set():
            time.sleep(1.0)
            cap.release()
            cap = _open_capture(target)
            ok, frame = cap.read()
        if self._stop.is_set():
            return
        h, w = frame.shape[:2]

        counter = LineCounter(line_from_frac(self._line_frac, w, h))
        rounds = RoundManager(round_seconds)
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        fps_ema = 0.0
        t_prev = time.time()

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                time.sleep(1.5)
                cap = _open_capture(target)
                continue

            if self._pending_line is not None:
                counter.set_line(line_from_frac(self._pending_line, w, h))
                self._line_frac = self._pending_line
                self._pending_line = None
            if self._pending_classes is not None:
                self._classes = self._pending_classes
                self._pending_classes = None
                counter.reset()
            if self._pending_threshold is not None:
                self._threshold = self._pending_threshold
                self._pending_threshold = None

            class_ids = [COCO[c] for c in self._classes if c in COCO] or None
            res = model.track(
                frame, persist=True, conf=conf, classes=class_ids,
                tracker="bytetrack.yaml", device=device, verbose=False,
            )[0]

            tracks = []
            if res.boxes is not None and res.boxes.id is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.int().cpu().tolist()
                cls = res.boxes.cls.int().cpu().tolist()
                for box, tid, c in zip(xyxy, ids, cls):
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    tracks.append({"id": tid, "cls": c, "name": model.names[c],
                                   "box": box, "center": (cx, cy)})
            counter.update(tracks)

            if rounds.expired():
                result = rounds.finalize(counter.count, counter.per_class)
                result["threshold"] = self._threshold
                if self.on_round_end:
                    try:
                        self.on_round_end(result)
                    except Exception as e:
                        print("[engine] erro no on_round_end:", e)
                counter.reset()
                rounds.next_round()

            draw_line(frame, counter.a, counter.b)
            draw_boxes(frame, tracks)
            draw_hud(frame, counter.count, rounds.remaining, rounds.round_id, dict(counter.per_class))
            ok2, buf = cv2.imencode(".jpg", frame, enc)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt

            with self._lock:
                if ok2:
                    self._jpeg = buf.tobytes()
                self._state = {
                    "ready": True,
                    "round_id": rounds.round_id,
                    "remaining_s": round(rounds.remaining, 1),
                    "round_seconds": round_seconds,
                    "count": counter.count,
                    "per_class": dict(counter.per_class),
                    "threshold": self._threshold,
                    "question": self._question,
                    "classes": list(self._classes),
                    "line_frac": list(self._line_frac),
                    "w": w, "h": h,
                    "fps": round(fps_ema, 1),
                    "updated_at": utc_now_iso(),
                }

        cap.release()
