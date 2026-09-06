"""OracleEngine: roda o oraculo numa thread e expoe frame anotado + estado.

Loop em background com o ciclo do jogo em FASES (ver PhasedRound):
  betting -> apostas abertas + contagem rodando
  running -> apostas encerradas, contagem continua
  pause   -> rodada liquidada, mostrando o resultado
Ao virar a rodada, o alvo X da proxima = contagem final desta.

Fluidez: o YOLO roda so a cada 'detect_every' frames; os quadros intermediarios
reaproveitam as ultimas deteccoes. Assim o video segue na taxa da camera (sem
travar por causa da inferencia) e a GPU alivia. O frame e reduzido (stream_width)
antes de virar JPEG, deixando o MJPEG mais leve no navegador.

Callbacks (disparados fora do lock):
  on_betting_close()            -> fecha as apostas do mercado
  on_round_end(result)          -> liquida o mercado com a contagem final
  on_new_round(prev_final, rid) -> abre o mercado da proxima rodada
"""

from __future__ import annotations

import threading
import time

import cv2

from .counting import LineCounter
from .overlay import draw_boxes, draw_hud, draw_line
from .round import PhasedRound, utc_now_iso
from .source import resolve_source

COCO = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
VEHICLES = ["car", "motorcycle", "bus", "truck"]

_PHASE_LABEL = {
    "betting": ("APOSTAS ABERTAS", (120, 220, 90)),
    "running": ("APOSTAS ENCERRADAS - contando", (60, 190, 245)),
    "pause": ("RODADA ENCERRADA - proxima em breve", (230, 170, 90)),
}


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
    def __init__(self, cfg: dict, on_round_end=None, on_betting_close=None, on_new_round=None):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.on_round_end = on_round_end
        self.on_betting_close = on_betting_close
        self.on_new_round = on_new_round
        self._lock = threading.Lock()
        self._jpeg = None
        self._state = {"ready": False}
        self._stop = threading.Event()

        self._classes = list(cfg.get("classes", VEHICLES))
        self._threshold = int(cfg.get("threshold", 5))
        self._question = ""
        self._line_frac = tuple(cfg.get("line_frac", (0.5, 0.42, 0.5, 0.85)))
        self._pending_line = None
        self._pending_classes = None
        self._pending_threshold = None
        self._jpeg_quality = int(cfg.get("jpeg_quality", 68))

    # ---- API thread-safe ----
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

    def _apply_pending(self, counter, w, h):
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

    # ---- loop principal ----
    def run(self):
        from ultralytics import YOLO

        target = resolve_source(str(self.cfg["source"]), int(self.cfg.get("max_height", 720)))
        model = YOLO(self.cfg.get("model", "yolo11s.pt"))
        device = self.cfg.get("device", 0)
        conf = float(self.cfg.get("conf", 0.3))
        imgsz = int(self.cfg.get("imgsz", 640))
        betting_s = float(self.cfg.get("betting_seconds", 30))
        round_s = float(self.cfg.get("round_seconds", 90))
        pause_s = float(self.cfg.get("pause_seconds", 15))
        detect_every = max(1, int(self.cfg.get("detect_every", 2)))
        stream_w = int(self.cfg.get("stream_width", 960))

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
        rounds = PhasedRound(betting_s, round_s, pause_s)
        prev_phase = rounds.phase
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        fps_ema = 0.0
        t_prev = time.time()
        fi = 0
        last_tracks = []

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                time.sleep(1.0)
                cap = _open_capture(target)
                continue

            self._apply_pending(counter, w, h)

            # roda o YOLO so a cada 'detect_every' frames -> video fluido, GPU leve
            fi += 1
            if fi % detect_every == 0 or not last_tracks:
                class_ids = [COCO[c] for c in self._classes if c in COCO] or None
                res = model.track(
                    frame, persist=True, conf=conf, classes=class_ids, imgsz=imgsz,
                    tracker="bytetrack.yaml", device=device, verbose=False,
                )[0]
                dets = []
                if res.boxes is not None and res.boxes.id is not None:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    ids = res.boxes.id.int().cpu().tolist()
                    cls = res.boxes.cls.int().cpu().tolist()
                    for box, tid, c in zip(xyxy, ids, cls):
                        cx = (box[0] + box[2]) / 2.0
                        cy = (box[1] + box[3]) / 2.0
                        dets.append({"id": tid, "cls": c, "name": model.names[c],
                                     "box": box, "center": (cx, cy)})
                last_tracks = dets
                if rounds.counting_active:
                    counter.update(dets)
            tracks = last_tracks

            # transicoes de fase
            phase = rounds.phase
            if phase != prev_phase:
                if prev_phase == "betting" and phase != "betting" and self.on_betting_close:
                    self._safe(self.on_betting_close)
                if phase == "pause" and prev_phase != "pause":
                    result = rounds.finalize(counter.count, counter.per_class)
                    result["threshold"] = self._threshold
                    if self.on_round_end:
                        self._safe(self.on_round_end, result)
                prev_phase = phase

            # fim do ciclo -> abre a proxima rodada (alvo = contagem desta)
            if rounds.cycle_over:
                prev_final = counter.count
                rounds.next_round()
                if self.on_new_round:
                    self._safe(self.on_new_round, prev_final, rounds.round_id)
                counter.reset()
                if self._pending_threshold is not None:
                    self._threshold = self._pending_threshold
                    self._pending_threshold = None
                prev_phase = rounds.phase

            # overlay (desenha as ultimas deteccoes em todo frame)
            draw_line(frame, counter.a, counter.b)
            draw_boxes(frame, tracks)
            draw_hud(frame, counter.count, rounds.phase_remaining, rounds.round_id, dict(counter.per_class))
            self._draw_phase(frame, rounds.phase)

            out = frame
            if stream_w and w > stream_w:
                out = cv2.resize(frame, (stream_w, int(h * stream_w / w)))
            ok2, buf = cv2.imencode(".jpg", out, enc)

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
                    "phase": rounds.phase,
                    "phase_remaining": round(rounds.phase_remaining, 1),
                    "remaining_s": round(rounds.phase_remaining, 1),
                    "betting_open": rounds.betting_open,
                    "counting_active": rounds.counting_active,
                    "betting_s": betting_s,
                    "round_s": round_s,
                    "pause_s": pause_s,
                    "count_remaining_s": round(max(0.0, round_s - rounds.elapsed), 1),
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

    @staticmethod
    def _safe(fn, *args):
        try:
            fn(*args)
        except Exception as e:
            print("[engine] erro em callback:", e)

    @staticmethod
    def _draw_phase(img, phase):
        text, color = _PHASE_LABEL.get(phase, (phase, (255, 255, 255)))
        w = img.shape[1]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = (w - tw) // 2
        cv2.rectangle(img, (x - 10, 96), (x + tw + 10, 96 + th + 14), (18, 18, 18), -1)
        cv2.putText(img, text, (x, 96 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
