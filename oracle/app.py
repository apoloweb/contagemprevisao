"""Loop principal do oraculo: ingere video, detecta, rastreia, conta, renderiza.

Junta as pecas:
  fonte de video -> YOLO (deteccao) -> ByteTrack (rastreio) ->
  LineCounter (contagem por cruzamento) -> RoundManager (rodadas) ->
  overlay + arquivos de saida (results.jsonl / state.json / annotated.mp4)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from .counting import LineCounter
from .overlay import draw_boxes, draw_hud, draw_line
from .round import RoundManager, utc_now_iso
from .source import resolve_source

# ids das classes COCO mais uteis
COCO = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
VEHICLES = ["car", "motorcycle", "bus", "truck"]


def _open_capture(target):
    if isinstance(target, str):
        cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(target)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass
    return cap


def _line_from_spec(spec, w, h):
    """'h' (horizontal), 'v' (vertical) ou 'x1,y1,x2,y2' (px ou fracoes <=1)."""
    if spec == "h":
        return ((0.0, h * 0.5), (float(w), h * 0.5))
    if spec == "v":
        return ((w * 0.5, 0.0), (w * 0.5, float(h)))
    vals = [float(v) for v in spec.split(",")]
    if max(vals) <= 1.0:  # coordenadas em fracao da imagem
        vals = [vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h]
    return ((vals[0], vals[1]), (vals[2], vals[3]))


def run(cfg: dict) -> None:
    from ultralytics import YOLO

    target = resolve_source(str(cfg["source"]), int(cfg.get("max_height", 720)))
    classes = cfg.get("classes", VEHICLES)
    class_ids = [COCO[c] for c in classes if c in COCO] or None

    model = YOLO(cfg.get("model", "yolo11n.pt"))
    device = cfg.get("device", 0)  # 0 = 1a GPU; "cpu" forca CPU

    outdir = Path(cfg.get("outdir", "runs"))
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"
    state_path = outdir / "state.json"

    cap = _open_capture(target)
    if not cap.isOpened():
        raise RuntimeError(f"Nao consegui abrir a fonte: {cfg['source']}")
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Fonte aberta, mas sem frames (stream offline?).")
    h, w = frame.shape[:2]
    print(f"[oraculo] fonte {w}x{h} | classes={classes} | rodada={cfg.get('round_seconds', 280)}s")

    line = _line_from_spec(str(cfg.get("line", "h")), w, h)
    counter = LineCounter(line)
    rounds = RoundManager(float(cfg.get("round_seconds", 280)))

    writer = None
    if cfg.get("record"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = float(cfg.get("out_fps") or 20.0)  # ~ taxa real de processamento
        writer = cv2.VideoWriter(str(outdir / "annotated.mp4"), fourcc, fps, (w, h))

    show = bool(cfg.get("show", False))
    conf = float(cfg.get("conf", 0.3))
    render = show or writer is not None
    last_state = 0.0
    start_wall = time.time()
    max_seconds = float(cfg.get("max_seconds") or 0)  # 0 = roda indefinidamente

    try:
        while True:
            if max_seconds and (time.time() - start_wall) >= max_seconds:
                print(f"[oraculo] parando (max_seconds={max_seconds:.0f}s atingido)")
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                # reconecta streams ao vivo que cairam
                print("[oraculo] frame perdido; reconectando em 2s...")
                cap.release()
                time.sleep(2.0)
                cap = _open_capture(target)
                continue

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
                    tracks.append({
                        "id": tid, "cls": c, "name": model.names[c],
                        "box": box, "center": (cx, cy),
                    })
            counter.update(tracks)

            # fim de rodada -> registra resultado e reinicia
            if rounds.expired():
                result = rounds.finalize(counter.count, counter.per_class)
                result["source"] = str(cfg["source"])
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"[rodada {result['round_id']}] final={result['final_count']} {dict(result['per_class'])}")
                counter.reset()
                rounds.next_round()

            # estado ao vivo ~1x/seg (consumido depois pelo frontend/mercado)
            now = time.time()
            if now - last_state >= 1.0:
                state = {
                    "round_id": rounds.round_id,
                    "remaining_s": round(rounds.remaining, 1),
                    "count": counter.count,
                    "per_class": dict(counter.per_class),
                    "updated_at": utc_now_iso(),
                }
                state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
                last_state = now

            if render:
                draw_line(frame, counter.a, counter.b)
                draw_boxes(frame, tracks)
                draw_hud(frame, counter.count, rounds.remaining, rounds.round_id, dict(counter.per_class))
                if writer is not None:
                    writer.write(frame)
                if show:
                    cv2.imshow("Oraculo - previsao", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
