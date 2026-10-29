"""Tests for the non-streaming JSON reply mode and the deploy_node dashboard.

These spin up real HTTP servers on ephemeral ports (no mocking), because the point
is to verify the wire format that ordinary chat front-ends depend on:

    POST /api/chat  {"messages": [...], "stream": false}  ->  {"reply": "..."}

That mode was added so a conventional OpenAI-shaped front-end -- including the
Gemini-generated `deploy_node.py` -- can talk to this model unmodified.
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
import urllib.request
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def chat_state():
    """A real, tiny GM-GPT checkpoint loaded into the chat server module."""
    import chat_server
    from gmllm.model import GMConfig, GMGPT
    from gmllm.tokenizer import ByteBPETokenizer

    torch.manual_seed(0)
    tok = ByteBPETokenizer.train(["the quick brown fox jumps over the lazy dog. " * 40],
                                 vocab_size=300)
    cfg = GMConfig(vocab_size=tok.vocab_size, d_model=32, n_head=2, n_layer=2,
                   ctx=64, dropout=0.0, n_limbs=2, limb_taps=8)
    model = GMGPT(cfg)
    chat_server.STATE.update({
        "model": model, "tok": tok, "cfg": cfg, "ban": set(), "stop_ids": set(),
        "atomic": True, "lock": threading.Lock(),
        "meta": {"checkpoint": "toy", "step": 0, "val_ppl": 9.0,
                 "atomic_markers": True, "banned_tokens": 0,
                 "params": sum(p.numel() for p in model.parameters())},
    })
    return chat_server


@pytest.fixture()
def live_server(chat_state):
    from http.server import ThreadingHTTPServer
    port = _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), chat_state.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()


def _post(url, payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read())


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read()


# ---------------------------------------------------------------- JSON mode --

def test_stream_false_returns_single_json_reply(live_server):
    status, data = _post(f"{live_server}/api/chat", {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8, "stream": False})
    assert status == 200
    assert set(data) >= {"reply", "tokens", "finish", "seconds"}
    assert isinstance(data["reply"], str)
    assert isinstance(data["tokens"], int)
    assert data["finish"] in ("stop", "length")


def test_stream_false_reply_is_not_an_fstring_of_a_count(live_server):
    """The whole point of the port: the reply comes from the model.

    The original deploy_node.py replied with
    f"Analysis for '{raw_msg}': Evaluated concept across {len(nodes)} nodes."
    i.e. a fixed template.  Two different prompts must not produce that shape.
    """
    _, a = _post(f"{live_server}/api/chat", {
        "messages": [{"role": "user", "content": "brown fox"}],
        "max_tokens": 6, "stream": False})
    assert "spatial topology nodes" not in a["reply"]
    assert "Analysis for" not in a["reply"]


def test_omitting_stream_still_returns_sse(chat_state, live_server):
    """Default must remain SSE -- existing front-ends depend on it."""
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 4}).encode()
    req = urllib.request.Request(f"{live_server}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        assert r.headers["Content-Type"].startswith("text/event-stream")
        text = r.read().decode()
    assert text.startswith("data: ")
    assert '"done": true' in text


def test_empty_messages_is_400(live_server):
    import urllib.error
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(f"{live_server}/api/chat", {"messages": [], "stream": False})
    assert e.value.code == 400


def test_status_reports_real_meta(live_server):
    status, raw = _get(f"{live_server}/api/status")
    s = json.loads(raw)
    assert status == 200 and s["ok"] is True
    assert s["params"] > 0 and s["vocab"] > 0
    assert "checkpoint" in s


# ------------------------------------------------------------- the dashboard --

def test_deploy_node_has_no_fabricated_hardware():
    """Audit guard: none of the invented upstream API may creep back into CODE.

    Checked via the AST (names actually referenced), not raw text -- the module
    docstring legitimately *documents* what was removed, and a text scan would
    flag its own audit trail.
    """
    import ast
    src = (ROOT / "deploy_node.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    referenced = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.alias):
            referenced.add(node.name.split(".")[-1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # catch the fabricated *module paths* used in import-like strings
            referenced.add(node.value)

    for banned in ("PixelHardwareBridge", "SpatialTopologyEngine",
                   "RecursiveSelfImprovementEngine",
                   "octotetrahedral_spatial_rsi",
                   "octotetrahedral_device_assistant"):
        assert banned not in referenced, (
            f"{banned} was fabricated by the generator and is still referenced in code")


def test_deploy_node_page_has_no_fake_controls():
    html = (ROOT / "deploy_node.html").read_text(encoding="utf-8")
    for banned in ("torch_on", "torch_off", "vibrate", "Battery:", "RSI Engine"):
        assert banned not in html, f"{banned} is a fabricated control"


def test_deploy_node_probe_reports_offline_cleanly():
    """probe_model must not raise when nothing is listening."""
    import deploy_node
    out = deploy_node.probe_model(_free_port(), timeout=1.0)
    assert out["ok"] is False and "error" in out


def test_deploy_node_proxies_chat_to_model(chat_state, live_server):
    """The dashboard's buffered JSON mode must forward to the real model server."""
    import deploy_node
    model_port = int(live_server.rsplit(":", 1)[1])
    deploy_node.STATE.update({"port": model_port, "timeout": 60,
                              "page": b"<html>ok</html>"})
    port = _free_port()
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", port), deploy_node.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        status, data = _post(f"http://127.0.0.1:{port}/api/chat",
                             {"messages": [{"role": "user", "content": "hello"}],
                              "max_tokens": 6, "stream": False})
        assert status == 200
        assert isinstance(data["reply"], str)
        assert "model_seconds" in data
    finally:
        srv.shutdown()
        srv.server_close()


def test_deploy_node_streams_sse_by_default(chat_state, live_server):
    """Regression: the default path must stream SSE, not buffer JSON.

    The first version of deploy_node.html spoke SSE but the dashboard answered
    with buffered JSON, so the page rendered "(empty)".  This locks the wire
    format the page depends on.
    """
    import deploy_node
    model_port = int(live_server.rsplit(":", 1)[1])
    deploy_node.STATE.update({"port": model_port, "timeout": 60,
                              "page": b"<html>ok</html>"})
    port = _free_port()
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", port), deploy_node.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        body = json.dumps({"messages": [{"role": "user", "content": "hello"}],
                           "max_tokens": 6}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            assert r.headers["Content-Type"].startswith("text/event-stream")
            text = r.read().decode()
        assert text.startswith("data: "), "page parses `data:` lines; got: " + text[:80]
        assert '"done": true' in text
    finally:
        srv.shutdown()
        srv.server_close()


def test_deploy_node_page_requests_streaming():
    """The page must not opt out of streaming."""
    html = (ROOT / "deploy_node.html").read_text(encoding="utf-8")
    assert '"stream": false' not in html
    assert "getReader()" in html, "page must read the stream incrementally"
