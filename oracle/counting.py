"""Contagem por cruzamento de linha a partir de rastros (tracks).

Cada objeto rastreado tem um id estável. Quando o segmento entre a posição
anterior e a atual do centro do objeto cruza a linha de contagem, contamos +1
uma única vez por id.
"""

from __future__ import annotations

from collections import defaultdict, deque


def _ccw(a, b, c) -> float:
    """Produto vetorial: sinal indica de que lado de a->b o ponto c está."""
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(p1, p2, a, b) -> bool:
    """True se o segmento p1->p2 cruza o segmento a->b (posição geral)."""
    d1 = _ccw(a, b, p1)
    d2 = _ccw(a, b, p2)
    d3 = _ccw(p1, p2, a)
    d4 = _ccw(p1, p2, b)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


class LineCounter:
    """Conta objetos que cruzam uma linha, sem contar o mesmo id duas vezes."""

    def __init__(self, line):
        self.a, self.b = line  # ((x1,y1),(x2,y2))
        self._prev_center: dict[int, tuple[float, float]] = {}
        self._last_seen: dict[int, int] = {}
        self._frame = 0
        self.counted: set[int] = set()
        self.count = 0
        self.per_class: dict[str, int] = defaultdict(int)
        self.events = deque(maxlen=100)

    def set_line(self, line) -> None:
        self.a, self.b = line

    def update(self, tracks) -> None:
        """tracks: lista de dicts com chaves id, name, center=(cx,cy)."""
        self._frame += 1
        for t in tracks:
            tid = t["id"]
            center = t["center"]
            self._last_seen[tid] = self._frame
            prev = self._prev_center.get(tid)
            self._prev_center[tid] = center
            if prev is None or tid in self.counted:
                continue
            if _segments_cross(prev, center, self.a, self.b):
                self.counted.add(tid)
                self.count += 1
                self.per_class[t["name"]] += 1
                direction = "A" if _ccw(self.a, self.b, center) < 0 else "B"
                self.events.append({"id": tid, "name": t["name"], "dir": direction})
        self._prune()

    def _prune(self, max_age: int = 150) -> None:
        """Evita crescimento infinito de memória em streams longos."""
        if len(self._prev_center) < 4000:
            return
        stale = [i for i, f in self._last_seen.items() if self._frame - f > max_age]
        for i in stale:
            self._prev_center.pop(i, None)
            self._last_seen.pop(i, None)
            self.counted.discard(i)

    def reset(self) -> None:
        """Zera a contagem no início de uma nova rodada (mantém os rastros)."""
        self.counted.clear()
        self.count = 0
        self.per_class = defaultdict(int)
        self.events.clear()
