"""CLI do oraculo de contagem.

Exemplos:
  # live de rua no YouTube, contando veiculos, com janela ao vivo
  python run_oracle.py --source "https://www.youtube.com/watch?v=XXXX" --line h --show

  # feed HLS publico direto (.m3u8), gravando um mp4 anotado
  python run_oracle.py --source "https://.../stream.m3u8" --record

  # webcam, contando pessoas
  python run_oracle.py --source 0 --classes person --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oracle.app import run


def load_yaml(path: str) -> dict:
    import yaml

    if path and Path(path).exists():
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {}


def main() -> None:
    p = argparse.ArgumentParser(description="Oraculo de contagem ao vivo")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", help="webcam(0), arquivo, .m3u8 ou pagina de live")
    p.add_argument("--model", help="ex.: yolo11n.pt (rapido) / yolo11s.pt (preciso)")
    p.add_argument("--classes", help="lista separada por virgula, ex: car,truck,bus")
    p.add_argument("--line", help="h, v ou x1,y1,x2,y2 (px ou fracoes <=1)")
    p.add_argument("--round-seconds", type=float, dest="round_seconds")
    p.add_argument("--max-seconds", type=float, dest="max_seconds",
                   help="para automaticamente apos N segundos (0 = infinito)")
    p.add_argument("--conf", type=float)
    p.add_argument("--device", help="0 (GPU) ou cpu")
    p.add_argument("--max-height", type=int, dest="max_height")
    p.add_argument("--outdir")
    p.add_argument("--show", action="store_true", default=None)
    p.add_argument("--record", action="store_true", default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    for k, v in vars(args).items():
        if k == "config" or v is None:
            continue
        if k == "classes":
            cfg[k] = [c.strip() for c in v.split(",") if c.strip()]
        else:
            cfg[k] = v

    if not cfg.get("source"):
        p.error("informe --source ou defina source no config.yaml")
    run(cfg)


if __name__ == "__main__":
    main()
