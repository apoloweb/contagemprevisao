"""Ciclo de vida da rodada: timer de tempo fixo (ex.: 4m40s = 280s)."""

from __future__ import annotations

import time
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RoundManager:
    def __init__(self, round_seconds: float):
        self.round_seconds = float(round_seconds)
        self.round_id = 1
        self._start = time.monotonic()
        self.started_at = utc_now_iso()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def remaining(self) -> float:
        return max(0.0, self.round_seconds - self.elapsed)

    def expired(self) -> bool:
        return self.remaining <= 0.0

    def finalize(self, count: int, per_class: dict) -> dict:
        """Fecha a rodada atual e devolve o resultado (nao abre a proxima)."""
        return {
            "round_id": self.round_id,
            "started_at": self.started_at,
            "ended_at": utc_now_iso(),
            "duration_s": round(self.round_seconds, 2),
            "final_count": int(count),
            "per_class": dict(per_class),
        }

    def next_round(self) -> None:
        self.round_id += 1
        self._start = time.monotonic()
        self.started_at = utc_now_iso()
