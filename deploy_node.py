"""deploy_node.py -- OctoTetrahedral node dashboard, backed by the REAL model.

This is a port of a `deploy_node.py` that was generated from the GitMonsters
`octotetrahedral-agi` repo by Gemini.  The original is preserved verbatim in
`docs/deploy_node_original.py` and audited in `DEPLOY.md`.  What changed and why:

  REMOVED (fabricated -- none of it exists in the upstream repo):
    * `PixelHardwareBridge` (battery / vibrate / torch via Termux)
    * `SpatialTopologyEngine`
    * `RecursiveSelfImprovementEngine`
  Upstream `octo_serve.py` exposes `/`, `/health`, `/tag`, `/analyze`,
  `/modules` -- and contains no hardware, battery, or Termux code at all.

  REMOVED (not intelligence -- it was an f-string):
    The original "AGI" path hashed the user's text into a 3x3 bit grid, called a
    graph function on it, then replied with an f-string containing a node COUNT.
    The reply did not depend on the model.  It is replaced with a real forward
    pass through the trained GM-GPT checkpoint.

  KEPT (the genuinely useful part):
    The server contract -- `POST /api/chat` with
    `{"messages": [{"role","content"}, ...]}` returning `{"reply": ...}`.
    That is the conventional OpenAI-shaped chat API, and it is a better interface
    than the SSE-only one the first build shipped.  It was wired into
    `chat_server.py` as `"stream": false`, so this file is a thin client of the
    real server rather than a second implementation of it.

Run (the chat server must already be up on --port):
    python chat_server.py --checkpoint runs/chat_physics_lm/best.pt --port 8765
    python deploy_node.py --port 8765
    # then open http://127.0.0.1:8766/

Honest note on output quality: the default checkpoint is 3.3M params, and at that
size the model produces locally-plausible word shapes but not coherent answers
(see RESULTS.md).  This dashboard does not hide that: it reports tokens/second
and shows the raw text, so a bad answer looks bad.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = None  # filled in main() from --page

STATE = {"port": 8765, "timeout": 180, "page": None, "last_error": None}

# ---------------------------------------------------------------------------
# the only upstream-derived component that is real and worth keeping: a health
# probe that talks to the actual model server instead of inventing telemetry.
# ---------------------------------------------------------------------------


def probe_model(port: int, timeout: float = 5.0) -> dict:
    """GET the chat server's /api/status, or report why it is unreachable."""
    url = f"http://127.0.0.1:{port}/api/status"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return {"ok": True, **json.loads(r.read())}
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "url": url}


def ask_model(messages: list[dict], port: int, timeout: float,
              max_tokens: int = 96) -> dict:
    """Send a conversation to the real model and return its reply."""
    body = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,          # <- the mode added for exactly this caller
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"ok": True, **json.loads(r.read())}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read()[:200]!r}"}
    except (urllib.error.URLError, OSError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def open_model_stream(messages: list[dict], port: int, timeout: float,
                      max_tokens: int = 96):
    """Open the model server's SSE stream and return the response object.

    The caller must stream and close it.  Returning the raw response (rather than
    re-reading it into memory) is what lets this dashboard pass tokens through as
    they are produced -- buffering here would defeat streaming entirely.
    """
    body = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        # no "stream": false  -> model server uses SSE by default
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, STATE["page"], "text/html; charset=utf-8")
        elif self.path == "/api/health":
            # real status of the real model, not a fabricated battery reading
            self._json({"dashboard": "ok", "model": probe_model(STATE["port"])})
        elif self.path == "/api/status":
            self._json(probe_model(STATE["port"]))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/chat":
            self._json({"error": "not found"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._json({"error": "bad json"}, 400)
            return

        messages = req.get("messages") or []
        messages = [{"role": str(m.get("role", "user")),
                     "content": str(m.get("content", ""))}
                    for m in messages if isinstance(m, dict)]
        if not messages:
            self._json({"error": "messages required"}, 400)
            return

        max_tokens = int(req.get("max_tokens", 96))

        # --- streaming pass-through (default) -----------------------------
        # The dashboard proxies the model server's SSE byte-for-byte, so tokens
        # appear as they are generated.  This is the path deploy_node.html uses.
        if req.get("stream", True) is not False:
            try:
                up = open_model_stream(messages, STATE["port"], STATE["timeout"],
                                       max_tokens)
            except urllib.error.HTTPError as e:
                self._json({"error": f"model HTTP {e.code}"}, 502)
                return
            except (urllib.error.URLError, OSError) as e:
                self._json({"error": f"model unreachable: {type(e).__name__}: {e}"}, 502)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                with up:
                    while True:
                        chunk = up.read(512)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        # --- buffered JSON mode -------------------------------------------
        t0 = time.time()
        out = ask_model(messages, STATE["port"], STATE["timeout"], max_tokens)
        if not out.get("ok"):
            self._json({"error": out.get("error", "model unreachable")}, 502)
            return
        self._json({"reply": out.get("reply", ""),
                    "tokens": out.get("tokens"),
                    "finish": out.get("finish"),
                    "model_seconds": out.get("seconds"),
                    "seconds": round(time.time() - t0, 2)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8766,
                   help="port to serve this dashboard on")
    p.add_argument("--model-port", type=int, default=8765,
                   help="port the chat server (real model) is listening on")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--page", default="deploy_node.html")
    args = p.parse_args()

    STATE.update({"port": args.model_port, "timeout": args.timeout})
    with open(args.page, "rb") as f:
        STATE["page"] = f.read()

    health = probe_model(args.model_port)
    if health.get("ok"):
        print(f"[deploy] model  : {args.model_port}  step {health.get('step')}  "
              f"{health.get('params', 0)/1e6:.2f}M params  "
              f"answer-token PPL {health.get('val_ppl'):.1f}")
    else:
        print(f"[deploy] WARNING: no model on {args.model_port} -- {health.get('error')}")
        print("[deploy] start it with:  python chat_server.py "
              "--checkpoint runs/chat_physics_lm/best.pt --port 8765")

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[deploy] open http://127.0.0.1:{args.port}/  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[deploy] stopped")


if __name__ == "__main__":
    main()
