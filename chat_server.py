#!/usr/bin/env python3
"""GM-GPT chat server -- stdlib only, CPU, streaming.

Serves the chat window at http://127.0.0.1:8765/ and streams replies token by
token over Server-Sent Events (POST /api/chat).

    python chat_server.py --checkpoint runs/chat/best.pt --port 8765

The generation loop reuses the model's KV cache (attention + fractional-mixer
history), applies repetition penalty / top-k / top-p, and stops on the chat
template's control markers.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F

from gmllm.chat import STOP_MARKERS, messages_to_prompt_with_budget, strip_stop
from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer

ROOT = Path(__file__).resolve().parent
STATE: dict = {"model": None, "tok": None, "cfg": None, "meta": {},
               "lock": threading.Lock(), "ban": set(), "stop_ids": set(),
               "atomic": False}


def load_checkpoint(path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GMConfig.from_dict(ckpt["config"])
    tok = ByteBPETokenizer.load(ckpt["tokenizer"])
    model = GMGPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Atomic control markers (single-token <|end|> etc.) change how the server
    # must behave: the marker is now a *reachable* prediction, so we stop on its
    # id rather than banning it.  See probe_atomic.py / CHAT.md.
    atomic = bool(tok.specials)
    stop_ids = {tok.id_for_special(m) for m in STOP_MARKERS if tok.id_for_special(m) >= 0}
    banned = set() if atomic else token_ban_list(tok)
    STATE.update({"model": model, "tok": tok, "cfg": cfg, "ban": banned,
                  "stop_ids": stop_ids, "atomic": atomic,
                  "meta": {"checkpoint": path, "step": ckpt.get("step"),
                           "val_ppl": ckpt.get("val_ppl"),
                           "atomic_markers": atomic,
                           "banned_tokens": len(banned),
                           "params": sum(p.numel() for p in model.parameters())}})


@torch.no_grad()
def stream_reply(prompt: str, temperature: float = 0.7, top_k: int = 40,
                 top_p: float = 0.95, max_tokens: int = 64,
                 rep_penalty: float = 1.15, ban: bool = True):
    """Yield ``(delta_text, n_tokens, finished)`` while generating a reply.

    ``finished`` is True only on the final yield, and only when the model chose
    to stop by emitting a control marker itself (rather than running out of
    ``max_tokens``).  That distinction is the whole point of the atomic-marker
    run, so it is reported rather than inferred by the client.
    """
    model, tok = STATE["model"], STATE["tok"]
    banned = STATE["ban"] if ban else set()
    stop_ids = STATE["stop_ids"]
    ids = tok.encode(prompt)
    if not ids:
        yield "", 0, False
        return
    idx = torch.tensor([ids], dtype=torch.long)
    past = None
    emitted = ""
    generated = 0
    recent: list[int] = []

    for _ in range(max_tokens):
        step_in = idx if past is None else idx[:, -1:]
        logits, _, _, past = model(step_in, use_cache=True, past=past)
        logits = logits[:, -1, :].float() / max(temperature, 1e-5)

        if rep_penalty != 1.0 and recent:
            seen = sorted(set(recent[-48:]))
            logits[:, seen] = torch.where(logits[:, seen] > 0,
                                          logits[:, seen] / rep_penalty,
                                          logits[:, seen] * rep_penalty)
        if banned:
            logits[:, sorted(banned)] = float("-inf")
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        if 0.0 < top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            probs = F.softmax(sl, dim=-1)
            cum = probs.cumsum(dim=-1)
            remove = cum > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sl = sl.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)

        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        nid = int(nxt.item())
        recent.append(nid)
        idx = torch.cat([idx, nxt], dim=1)
        generated += 1

        # Atomic markers: the model can emit the stop token directly.
        if nid in stop_ids:
            yield "", generated, True
            return

        raw = tok.decode(idx[0, len(ids):].tolist())
        stopped = strip_stop(raw)
        if stopped != raw:                      # hit <|end|> / <|user|> ...
            if stopped.startswith(emitted):
                delta = stopped[len(emitted):]
                if delta:
                    yield delta, generated, False
            yield "", generated, True
            return
        if raw.endswith("\ufffd"):              # incomplete UTF-8, wait a token
            continue
        display = _hold_partial_marker(raw)      # don't leak "<|en" while waiting
        if display.startswith(emitted):
            delta = display[len(emitted):]
            if delta:
                emitted = display
                yield delta, generated, False
        elif raw.startswith(emitted):            # marker prefix turned out to be text
            delta = raw[len(emitted):]
            if delta:
                emitted = raw
                yield delta, generated, False


def _hold_partial_marker(text: str) -> str:
    """Withhold a trailing prefix of a control marker until it resolves.

    Prevents ``<|en`` flashing in the UI before ``<|end|>`` arrives (which
    would then be impossible to retract with append-only streaming).
    """
    for marker in STOP_MARKERS:
        for k in range(1, len(marker)):
            if text.endswith(marker[:k]):
                return text[:-k]
    return text


def token_ban_list(tok, allow_newline: bool = True) -> set[int]:
    """Token ids to forbid while sampling a reply.

    Some byte-BPE tokens are *pieces of the chat control markers* (``<``, ``|``,
    ``>``, ``end``, ``ant``, ``si``, ``us`` ...).  At this model size the
    log-probabilities of a multi-token marker spread over its pieces, so the
    pieces individually lose the argmax race to whole English words such as
    ``the`` -- while their *sum* still wins.  The result is the marker soup in
    the SFT samples: the model is reaching for ``<|end|>`` but can only emit it
    one fragment at a time, and the sampling order scrambles it.

    Banning the pieces makes the marker unreachable mid-reply, so the model
    simply cannot start one -- it is also prevented from *ending* a reply, so a
    budget of a few hundred tokens (see ``--max-tokens``) is the natural stop.
    This is a decoding fix, not a training fix: everything below is measured
    with it enabled.
    """
    banned = set()
    markers = STOP_MARKERS
    pieces = set()
    for marker in markers:
        ids = tok.encode(marker)
        if len(ids) > 1:                 # multi-token marker: its pieces are unsafe
            pieces.update(ids)
    banned |= pieces
    if not allow_newline:
        banned |= set(tok.encode("\n"))
    return banned


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = (ROOT / "chat" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page)
        elif self.path == "/api/status":
            self._send_json({"ok": True, **STATE["meta"],
                             "vocab": STATE["tok"].vocab_size})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/chat":
            self._send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send_json({"error": "bad json"}, 400)
            return

        messages = req.get("messages") or []
        if not isinstance(messages, list) or not messages:
            self._send_json({"error": "messages required"}, 400)
            return
        try:
            temperature = min(max(float(req.get("temperature", 0.7)), 0.0), 2.0)
            top_k = int(req.get("top_k", 40))
            top_p = min(max(float(req.get("top_p", 0.95)), 0.05), 1.0)
            max_tokens = min(max(int(req.get("max_tokens", 64)), 1), 512)
            rep = min(max(float(req.get("rep_penalty", 1.15)), 1.0), 2.0)
        except (TypeError, ValueError):
            self._send_json({"error": "temperature/top_k/top_p/max_tokens/rep_penalty must be numbers"}, 400)
            return
        ban = bool(req.get("ban_marker_tokens", True))

        cfg = STATE["cfg"]
        budget = max(32, cfg.ctx - max_tokens - 8)
        prompt = messages_to_prompt_with_budget(
            messages, STATE["tok"].encode, budget)

        # Non-streaming mode: `"stream": false` (or accept: application/json)
        # returns a single OpenAI-shaped JSON body {"reply": ...}.  This exists
        # so ordinary chat front-ends -- including the Gemini-generated
        # deploy_node.py page -- can point at this model unmodified.  Same
        # generator loop as SSE; only the transport differs.
        if req.get("stream") is False:
            t0 = time.time()
            out, n, finished = [], 0, False
            try:
                with STATE["lock"]:
                    for delta, n, finished in stream_reply(prompt, temperature, top_k,
                                                           top_p, max_tokens, rep, ban):
                        if delta:
                            out.append(delta)
            except (BrokenPipeError, ConnectionResetError):
                pass
            self._send_json({"reply": "".join(out), "tokens": n,
                             "finish": "stop" if finished else "length",
                             "seconds": round(time.time() - t0, 2)})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send(payload: dict):
            self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        t0 = time.time()
        n = 0
        finished = False
        try:
            with STATE["lock"]:
                for delta, n, finished in stream_reply(prompt, temperature, top_k,
                                                       top_p, max_tokens, rep, ban):
                    if delta:
                        send({"t": delta})
            send({"done": True, "tokens": n,
                  "seconds": round(time.time() - t0, 2),
                  "finish": "stop" if finished else "length",
                  "tok_s": int(n / max(time.time() - t0, 1e-6))})
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/chat/best.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    load_checkpoint(args.checkpoint)
    m = STATE["meta"]
    mode = ("atomic markers — the model can stop itself" if m["atomic_markers"]
            else f"{m['banned_tokens']} marker-piece tokens banned")
    print(f"[chat] loaded {args.checkpoint} | {m['params']/1e6:.2f}M params "
          f"| step {m['step']} | answer-token PPL {m['val_ppl']} | {mode}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[chat] open http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[chat] stopped")


if __name__ == "__main__":
    main()
