"""Renderiza o overlay sobre o frame: caixas, linha de contagem e HUD.

Visual inspirado no previsao.io: caixas ciano, linha rosa, contador grande no
canto superior esquerdo e timer no canto superior direito.
"""

from __future__ import annotations

import cv2

# cores em BGR
CYAN = (255, 255, 0)
GREEN = (80, 220, 120)
PINK = (200, 60, 255)
WHITE = (245, 245, 245)
DARK = (18, 18, 18)
RED = (60, 60, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(img, x, y, w, h, alpha: float = 0.55) -> None:
    """Retangulo escuro translucido para dar contraste ao texto."""
    h_img, w_img = img.shape[:2]
    x2, y2 = min(x + w, w_img), min(y + h, h_img)
    x, y = max(x, 0), max(y, 0)
    if x2 <= x or y2 <= y:
        return
    roi = img[y:y2, x:x2]
    overlay = roi.copy()
    overlay[:] = DARK
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def draw_boxes(img, tracks) -> None:
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), CYAN, 2)
        label = f'{t["name"]} #{t["id"]}'
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), CYAN, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), FONT, 0.5, DARK, 1, cv2.LINE_AA)


def draw_line(img, a, b) -> None:
    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), PINK, 3, cv2.LINE_AA)


def draw_hud(img, count, remaining_s, round_id, per_class=None) -> None:
    h, w = img.shape[:2]

    # contador (canto superior esquerdo)
    _panel(img, 12, 12, 250, 74)
    cv2.putText(img, "CONTAGEM ATUAL", (24, 36), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, str(count), (24, 78), FONT, 1.3, GREEN, 3, cv2.LINE_AA)

    # timer (canto superior direito)
    mm, ss = divmod(int(remaining_s), 60)
    _panel(img, w - 190, 12, 178, 74)
    cv2.putText(img, "TEMPO", (w - 176, 36), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    color = RED if remaining_s <= 15 else WHITE
    cv2.putText(img, f"{mm:02d}:{ss:02d}", (w - 176, 78), FONT, 1.2, color, 3, cv2.LINE_AA)

    # rodape: rodada + contagem por classe
    parts = [f"Rodada {round_id}"]
    if per_class:
        parts += [f"{k}: {v}" for k, v in per_class.items()]
    footer = "   ".join(parts)
    _panel(img, 12, h - 40, min(w - 24, 24 + len(footer) * 9), 30)
    cv2.putText(img, footer, (20, h - 19), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
