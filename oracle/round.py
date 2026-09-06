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


class PhasedRound:
    """Rodada em fases, no modelo do jogo:

      [betting]  0 .. betting_s      -> apostas ABERTAS + contagem rodando
      [running]  betting_s .. round_s-> apostas ENCERRADAS, contagem continua
      [pause]    round_s .. +pause_s -> rodada liquidada, mostrando resultado

    Ao fim da pausa comeca a proxima rodada (o alvo X da proxima = contagem
    final desta).
    """

    def __init__(self, betting_s: float, round_s: float, pause_s: float):
        self.betting_s = float(betting_s)
        self.round_s = float(round_s)      # duracao total da contagem
        self.pause_s = float(pause_s)
        self.round_id = 1
        self._start = time.monotonic()
        self.started_at = utc_now_iso()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def phase(self) -> str:
        e = self.elapsed
        if e < self.betting_s:
            return "betting"
        if e < self.round_s:
            return "running"
        return "pause"

    @property
    def phase_remaining(self) -> float:
        e = self.elapsed
        if e < self.betting_s:
            return self.betting_s - e
        if e < self.round_s:
            return self.round_s - e
        return max(0.0, self.round_s + self.pause_s - e)

    @property
    def counting_active(self) -> bool:
        return self.elapsed < self.round_s

    @property
    def betting_open(self) -> bool:
        return self.elapsed < self.betting_s

    @property
    def cycle_over(self) -> bool:
        return self.elapsed >= self.round_s + self.pause_s

    def finalize(self, count: int, per_class: dict) -> dict:
        return {
            "round_id": self.round_id,
            "started_at": self.started_at,
            "ended_at": utc_now_iso(),
            "duration_s": round(self.round_s, 2),
            "final_count": int(count),
            "per_class": dict(per_class),
        }

    def next_round(self) -> None:
        self.round_id += 1
        self._start = time.monotonic()
        self.started_at = utc_now_iso()
