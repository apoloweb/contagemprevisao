"""Motor de mercado de previsao com moeda virtual (livro de ofertas).

Modelo (binario por rodada): a pergunta e "a contagem final vai passar de X?".
- Acao OVER paga R$1,00 (100 centavos) se contagem_final > X, senao 0.
- Acao UNDER paga R$1,00 se contagem_final <= X, senao 0.
Como OVER+UNDER = R$1,00, o casamento acontece por "minting": quando o melhor
lance de OVER + melhor lance de UNDER >= 100c, a casa cunha o par e entrega uma
acao pra cada lado. Precos em centavos (1..99) ~ probabilidade implicada.

Tudo em centavos internamente. Saldo inicial padrao: R$1.000,00.
"""

from __future__ import annotations

import itertools
import threading
import time
from datetime import datetime, timezone

_oid = itertools.count(1)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Market:
    """Um mercado por rodada: dois livros de lances (OVER e UNDER)."""

    def __init__(self, round_id: int, threshold: int, question: str):
        self.round_id = round_id
        self.threshold = threshold
        self.question = question
        self.open = True
        self.over_bids: list[dict] = []   # lances de compra de OVER
        self.under_bids: list[dict] = []  # lances de compra de UNDER
        self.trades: list[dict] = []
        self.last_over = None

    def add(self, order: dict) -> None:
        book = self.over_bids if order["side"] == "over" else self.under_bids
        book.append(order)
        book.sort(key=lambda o: (-o["price"], o["ts"]))  # melhor preco primeiro

    def match(self, on_fill) -> list:
        fills = []
        while self.over_bids and self.under_bids:
            ob, ub = self.over_bids[0], self.under_bids[0]
            if ob["price"] + ub["price"] < 100:
                break  # nao cruza: sem par a cunhar
            qty = min(ob["left"], ub["left"])
            p_over, p_under = ob["price"], ub["price"]
            on_fill(ob, ub, qty, p_over, p_under)
            ob["left"] -= qty
            ub["left"] -= qty
            self.last_over = p_over
            self.trades.append({"ts": time.time(), "qty": qty, "p_over": p_over, "p_under": p_under})
            fills.append({"qty": qty, "p_over": p_over, "p_under": p_under})
            if ob["left"] == 0:
                self.over_bids.pop(0)
            if ub["left"] == 0:
                self.under_bids.pop(0)
        return fills

    @staticmethod
    def _levels(book: list) -> list:
        agg: dict = {}
        for o in book:
            if o["left"] > 0:
                agg[o["price"]] = agg.get(o["price"], 0) + o["left"]
        return sorted(({"price": p, "qty": q} for p, q in agg.items()), key=lambda x: -x["price"])

    def view(self) -> dict:
        over = self._levels(self.over_bids)
        under = self._levels(self.under_bids)
        return {
            "round_id": self.round_id,
            "threshold": self.threshold,
            "question": self.question,
            "open": self.open,
            "over_bids": over[:8],
            "under_bids": under[:8],
            "best_over": over[0]["price"] if over else None,
            "best_under": under[0]["price"] if under else None,
            "last_over": self.last_over,
            "trades": [{"qty": t["qty"], "p_over": t["p_over"]} for t in self.trades[-14:]],
        }


class Exchange:
    def __init__(self, start_balance: int = 100000):  # R$1.000,00 em centavos
        self.lock = threading.RLock()
        self.start_balance = start_balance
        self.users: dict = {}
        self.market = None
        self.history: list = []
        self._ensure_user("you", "Voce")

    def _ensure_user(self, uid: str, name=None, balance=None) -> dict:
        if uid not in self.users:
            self.users[uid] = {
                "name": name or uid,
                "balance": self.start_balance if balance is None else balance,
                "reserved": 0,
                "positions": {"over": 0, "under": 0},
            }
        return self.users[uid]

    def open_market(self, round_id: int, threshold: int, question: str) -> "Market":
        with self.lock:
            self.market = Market(round_id, threshold, question)
            for u in self.users.values():          # posicoes sao por rodada
                u["positions"] = {"over": 0, "under": 0}
                u["balance"] += u["reserved"]       # devolve o que estava reservado
                u["reserved"] = 0
            return self.market

    def place(self, uid: str, side: str, price: int, qty: int) -> dict:
        with self.lock:
            u = self._ensure_user(uid)
            m = self.market
            if m is None or not m.open:
                return {"error": "mercado fechado"}
            if side not in ("over", "under"):
                return {"error": "lado invalido"}
            price = max(1, min(99, int(price)))
            qty = int(qty)
            if qty <= 0:
                return {"error": "quantidade invalida"}
            cost = price * qty
            if u["balance"] < cost:
                return {"error": "saldo insuficiente"}
            u["balance"] -= cost      # reserva o custo maximo
            u["reserved"] += cost
            order = {"id": next(_oid), "uid": uid, "side": side,
                     "price": price, "qty": qty, "left": qty, "ts": time.time()}
            m.add(order)
            fills = m.match(self._on_fill)
            return {"ok": True, "order_id": order["id"], "fills": fills}

    def _on_fill(self, ob: dict, ub: dict, qty: int, p_over: int, p_under: int) -> None:
        for order, side, fill_price in ((ob, "over", p_over), (ub, "under", p_under)):
            u = self.users[order["uid"]]
            reserved_here = order["price"] * qty          # reservamos ao preco do lance
            refund = (order["price"] - fill_price) * qty  # 0 aqui (fill = lance)
            u["reserved"] -= reserved_here
            u["balance"] += refund
            u["positions"][side] += qty

    def _refund_open_orders(self) -> None:
        m = self.market
        if not m:
            return
        for book in (m.over_bids, m.under_bids):
            for o in book:
                if o["left"] > 0:
                    u = self.users[o["uid"]]
                    back = o["price"] * o["left"]
                    u["reserved"] -= back
                    u["balance"] += back
                    o["left"] = 0

    def settle(self, final_count: int):
        with self.lock:
            m = self.market
            if not m:
                return None
            m.open = False
            self._refund_open_orders()
            over_wins = final_count > m.threshold
            summary = {
                "round_id": m.round_id, "threshold": m.threshold,
                "final_count": final_count, "winner": "over" if over_wins else "under",
                "at": _now(), "you_payout": 0,
            }
            for uid, u in self.users.items():
                win_shares = u["positions"]["over"] if over_wins else u["positions"]["under"]
                payout = win_shares * 100
                u["balance"] += payout
                if uid == "you":
                    summary["you_payout"] = payout
            self.history.append(summary)
            return summary

    def snapshot(self, uid: str = "you") -> dict:
        with self.lock:
            u = self._ensure_user(uid)
            return {
                "you": {
                    "name": u["name"],
                    "balance": u["balance"],
                    "reserved": u["reserved"],
                    "positions": dict(u["positions"]),
                },
                "market": self.market.view() if self.market else None,
                "history": self.history[-6:],
            }
