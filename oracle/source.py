"""Resolve a fonte de vídeo para algo que o OpenCV consiga abrir.

Aceita:
  - índice de webcam ................ "0", "1"
  - arquivo local ................... "rua.mp4"
  - stream HLS/RTSP/mp4 direto ...... "https://.../stream.m3u8", "rtsp://..."
  - página de live (YouTube etc.) ... resolvida via yt-dlp para o manifesto HLS
"""

from __future__ import annotations

import os


def _is_direct_media(url: str) -> bool:
    low = url.lower().split("?")[0]
    return (
        low.startswith("rtsp://")
        or low.startswith("rtmp://")
        or low.endswith(".m3u8")
        or low.endswith(".mp4")
        or low.endswith(".ts")
    )


def resolve_youtube(url: str, max_height: int = 720) -> str:
    """Extrai a URL direta do stream (HLS) de uma página de live via yt-dlp.

    Lives do YouTube nao tem formato "best" combinado; entregam faixas HLS
    (m3u8) e/ou adaptativas. O OpenCV/FFmpeg le bem uma media playlist m3u8,
    entao escolhemos a melhor faixa HLS com video, de altura <= max_height.
    """
    import yt_dlp  # import tardio: só carrega quando precisa

    ydl_opts = {"quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    formats = info.get("formats") or []

    def height(f) -> int:
        return f.get("height") or 0

    def is_hls(f) -> bool:
        return "m3u8" in (f.get("protocol") or "")

    # formatos que carregam video e tem URL utilizavel
    vids = [f for f in formats if f.get("vcodec") not in (None, "none") and f.get("url")]
    hls = [f for f in vids if is_hls(f)]

    candidates = [f for f in hls if height(f) <= max_height] or hls or vids
    if candidates:
        candidates.sort(key=height)
        return candidates[-1]["url"]  # a maior altura permitida

    # fallback: manifesto/URL direta do proprio info
    if info.get("manifest_url"):
        return info["manifest_url"]
    if info.get("url"):
        return info["url"]
    raise RuntimeError(f"Não consegui resolver um stream de: {url}")


def resolve_source(src: str, max_height: int = 720):
    """Devolve algo aceito por cv2.VideoCapture (int, caminho ou URL)."""
    # webcam por índice
    if src.isdigit():
        return int(src)

    # arquivo local
    if os.path.exists(src):
        return src

    low = src.lower()

    # páginas de live conhecidas -> yt-dlp
    if "youtube.com" in low or "youtu.be" in low or "twitch.tv" in low:
        return resolve_youtube(src, max_height)

    # stream/arquivo direto
    if _is_direct_media(src):
        return src

    # último recurso: tenta yt-dlp (suporta centenas de sites), senão devolve cru
    try:
        return resolve_youtube(src, max_height)
    except Exception:
        return src
