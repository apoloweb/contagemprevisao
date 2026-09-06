"""Servidor do app de mercado de previsao (oraculo + mercado + web).

Junta tudo:
  - OracleEngine roda numa thread (ingere HLS, detecta, conta, gerencia rodadas)
  - Exchange (livro de ofertas) com moeda virtual, liquidado ao fim de cada rodada
  - bots de liquidez mantem o livro sempre negociavel
  - servidor HTTP (stdlib) serve o video (MJPEG), a API JSON e a pagina web

Uso:
  python server.py --source "https://www.youtube.com/watch?v=<ID>" --port 8000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from market import Exchange
from oracle.engine import OracleEngine, VEHICLES

WEB_DIR = Path(__file__).parent / "web"

# ------------------------------------------------------------------ helpers

def label_for(classes) -> str:
    if list(classes) == ["person"]:
        return "pessoas"
    return "passagens"


def question_for(x: int, classes) -> str:
    return f"Vai passar de {x} {label_for(classes)}?"


def compute_fair(c: int, x: int, rem: float, rs: float) -> float:
    """Probabilidade justa de OVER (contagem_final > x). Contagem so cresce."""
    if c > x:
        return 0.985  # ja passou: OVER praticamente certo
    elapsed = max(rs - rem, 1.0)
    rate = c / elapsed
    expected_more = rate * max(rem, 0.0)
    needed = (x - c) + 0.5
    if expected_more <= 0:
        return 0.03 if rem < rs * 0.5 else 0.15
    z = (expected_more - needed) / max(1.0, math.sqrt(expected_more + 1.0))
    return min(0.97, max(0.03, 1.0 / (1.0 + math.exp(-z))))


# ------------------------------------------------------------------ app

class App:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.exchange = Exchange()
        # bots com saldo alto para nunca faltar liquidez
        for bot in ("MM1", "MM2", "T1", "T2"):
            self.exchange._ensure_user(bot, bot, balance=10**9)

        self.classes0 = cfg.get("classes", VEHICLES)
        self.x0 = int(cfg.get("threshold", 3))
        q0 = question_for(self.x0, self.classes0)

        self.engine = OracleEngine(
            cfg,
            on_round_end=self._on_round_end,
            on_betting_close=self._on_betting_close,
            on_new_round=self._on_new_round,
        )
        self.exchange.open_market(1, self.x0, q0)
        self.engine.set_threshold(self.x0)
        self.engine.set_question(q0)

        self._stop = threading.Event()

    def start(self):
        self.engine.start()
        threading.Thread(target=self._bots_loop, daemon=True).start()

    def stop(self):
        self._stop.set()
        self.engine.stop()

    def _on_betting_close(self):
        m = self.exchange.market
        if m:
            m.close_betting()

    def _on_round_end(self, result: dict):
        # as apostas ja encerraram na fase 'running'; aqui so liquida
        self.exchange.settle(int(result["final_count"]))

    def _on_new_round(self, prev_final: int, round_id: int):
        st = self.engine.get_state()
        classes = st.get("classes", self.classes0)
        next_x = max(1, int(prev_final))       # alvo da proxima = contagem desta
        q = question_for(next_x, classes)
        self.exchange.open_market(round_id, next_x, q)
        self.engine.set_threshold(next_x)
        self.engine.set_question(q)

    def _bots_loop(self):
        while not self._stop.is_set():
            try:
                st = self.engine.get_state()
                m = self.exchange.market
                if st.get("ready") and m and m.open and m.betting_open:
                    x = m.threshold
                    c = int(st.get("count", 0))
                    rem = float(st.get("remaining_s", 0))
                    rs = float(st.get("round_seconds", 60))
                    p = compute_fair(c, x, rem, rs)
                    po = min(98, max(2, round(p * 100)))
                    pu = 100 - po
                    # profundidade: lances de mercado dos dois lados (com spread)
                    self.exchange.place("MM1", "over", max(1, po - 2), random.randint(3, 10))
                    self.exchange.place("MM2", "under", max(1, pu - 2), random.randint(3, 10))
                    # tomador: cruza o spread ~70% das vezes para gerar negocios
                    if random.random() < 0.7:
                        v = m.view()
                        if random.random() < p and v["best_under"]:
                            self.exchange.place("T1", "over", min(99, 100 - v["best_under"]), random.randint(1, 4))
                        elif v["best_over"]:
                            self.exchange.place("T2", "under", min(99, 100 - v["best_over"]), random.randint(1, 4))
            except Exception as e:
                print("[bots] erro:", e)
            time.sleep(2.5)

    # ------------- acoes vindas da API -------------
    def state_json(self) -> dict:
        st = self.engine.get_state()
        snap = self.exchange.snapshot("you")
        return {"engine": st, "you": snap["you"], "market": snap["market"], "history": snap["history"]}

    def place_order(self, side: str, qty: int, price=None) -> dict:
        m = self.exchange.market
        if not m or not m.open:
            return {"error": "mercado fechado"}
        if price is None:  # ordem "a mercado": cruza o melhor preco do outro lado
            v = m.view()
            if side == "over":
                price = 100 - (v["best_under"] or 50)
            else:
                price = 100 - (v["best_over"] or 50)
        return self.exchange.place("you", side, int(price), int(qty))

    def reset_wallet(self) -> dict:
        with self.exchange.lock:
            u = self.exchange._ensure_user("you")
            u["balance"] = self.exchange.start_balance
            u["reserved"] = 0
            u["positions"] = {"over": 0, "under": 0}
        return {"ok": True}


# ------------------------------------------------------------------ HTTP

def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass  # silencia log de acesso

        def _send_json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n <= 0:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except Exception:
                return {}

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/" or path == "/index.html":
                self._serve_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send_json(app.state_json())
            elif path == "/stream.mjpg":
                self._stream_mjpeg()
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            path = self.path.split("?")[0]
            data = self._read_json()
            if path == "/api/order":
                self._send_json(app.place_order(data.get("side"), data.get("qty", 1), data.get("price")))
            elif path == "/api/line":
                app.engine.set_line_frac(data["x1"], data["y1"], data["x2"], data["y2"])
                self._send_json({"ok": True})
            elif path == "/api/classes":
                app.engine.set_classes(data.get("classes", []))
                self._send_json({"ok": True})
            elif path == "/api/reset":
                self._send_json(app.reset_wallet())
            else:
                self._send_json({"error": "not found"}, 404)

        def _serve_file(self, fpath: Path, ctype: str):
            try:
                body = fpath.read_bytes()
            except FileNotFoundError:
                self._send_json({"error": "arquivo nao encontrado"}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream_mjpeg(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while not app._stop.is_set():
                    jpeg = app.engine.get_jpeg()
                    if jpeg is None:
                        time.sleep(0.1)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.066)  # ~15 fps
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass

    return Handler


def load_cfg(args) -> dict:
    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    for k, v in vars(args).items():
        if k in ("config", "port") or v is None:
            continue
        if k == "classes":
            cfg[k] = [c.strip() for c in v.split(",") if c.strip()]
        else:
            cfg[k] = v
    cfg.setdefault("betting_seconds", 30)
    cfg.setdefault("round_seconds", 90)
    cfg.setdefault("pause_seconds", 15)
    cfg.setdefault("model", "yolo11s.pt")
    return cfg


def main():
    p = argparse.ArgumentParser(description="App de mercado de previsao (oraculo + mercado + web)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", help="webcam(0), arquivo, .m3u8 ou pagina de live")
    p.add_argument("--model")
    p.add_argument("--classes", help="ex: car,truck,bus,motorcycle ou person")
    p.add_argument("--betting-seconds", type=float, dest="betting_seconds")
    p.add_argument("--round-seconds", type=float, dest="round_seconds")
    p.add_argument("--pause-seconds", type=float, dest="pause_seconds")
    p.add_argument("--threshold", type=int)
    p.add_argument("--conf", type=float)
    p.add_argument("--device")
    p.add_argument("--max-height", type=int, dest="max_height")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    cfg = load_cfg(args)
    if not cfg.get("source"):
        p.error("informe --source ou defina 'source' no config.yaml")

    app = App(cfg)
    app.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    server.daemon_threads = True
    print(f"[server] http://127.0.0.1:{args.port}  (fonte: {cfg['source']})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
