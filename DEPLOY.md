# `deploy_node.py` — audit of a Gemini-generated artifact, and the port

You generated a dashboard from the GitMonsters `octotetrahedral-agi` repo with Gemini.
This document records what was checked, what was wrong, and what was kept.

**Verdict: the UI shell is fine, the "AGI" layer is a keyword router rather than a model, and
the script cannot run *on this machine* as generated.** The one genuinely useful idea — a
conventional JSON chat API — was real and has been wired into the real model. Everything else
was replaced with the model we trained.

> **Correction (see `docs/PHONE_BUILD.md`).** This document originally said the imported
> modules were **invented/fabricated** and "do not exist". **That was wrong.** They exist on
> the user's Pixel 7a at `~/octotetrahedral-agi/`, they are implemented, and
> `PixelHardwareBridge.get_battery()` was verified returning a live reading
> (`Battery: 29% | Status: CHARGING | Health: GOOD`) over `adb forward`.
>
> What remains true is narrower and still important: **they are not in the public repo**, so
> the generated script cannot run against a public clone — which is exactly what happened
> here. The failure is a *distribution* gap, not an invention. Read §1 with that correction
> in mind; §2 stands as written (the router is real, and there is no language model in the
> phone build either). The honest one-line description of the phone build is: **a working
> Android device-control dashboard with a keyword router instead of a language model.**

---

## 1. The generator imported API that is not in the public repo

The generated `server.py` begins:

```python
from octotetrahedral_spatial_rsi import SpatialTopologyEngine, RecursiveSelfImprovementEngine
from octotetrahedral_device_assistant import PixelHardwareBridge
```

**Neither module is in the public repo.** Measured across the repo cache:

```
=== do the two imported modules exist anywhere in octotetrahedral-agi? ===
(no files matching spatial_rsi | device_assistant | spatial | hardware)
=== any file defining these classes? ===
(no file defines SpatialTopologyEngine | RecursiveSelfImprovementEngine | PixelHardwareBridge)
=== does upstream have ANY hardware/battery/termux code? ===
(only PyTorch `import torch` hits; no termux, no battery, no vibrate)
```

So `python3 deploy_node.py` creates the files and prints `[OK]`, and then
`python3 server.py` fails immediately with `ModuleNotFoundError`. **The script never tested
its own success case** — the same shape of claim as `TRANSCENDPLEXITY`'s missing `tests/` and
`benchmarks/`. That defect is real and is the reason this port exists.

But note the *correct* cause: the modules live on the **device**, not in the repo. The
generator assumed a deployment target (a Pixel running Termux) that a public clone does not
provide. See `docs/PHONE_BUILD.md` §2 for how they were recovered and verified.

Upstream `octo_serve.py` is a real FastAPI server, and its actual routes are:

```
@app.get("/")            @app.get("/health")     @app.get("/modules")
@app.post("/tag")        @app.post("/tag/batch") @app.post("/analyze")
```

No battery endpoint, no torch endpoint, no haptics — in the **public** repo. The
`PixelHardwareBridge` surface (`get_battery()`, `vibrate()`, `control_torch()`) is real code
that exists only on the device. It is Termux-specific (`termux-battery-status`) and cannot
work on Windows, which is why this port does not attempt to ship it.

## 2. The "intelligence" is an f-string

The generated `server.py`'s catch-all branch:

```python
grid = [[ord(c) % 2 for c in (latest_msg + "123456789")[i:i+3]] for i in range(0, 9, 3)]
nodes = spatial_engine.extract_spatial_graph(grid)
reply = f"Analysis for '{raw_msg}': Evaluated concept across {len(nodes)} spatial topology nodes on a 3x3 manifold."
```

The reply is a **string template**. Its only dynamic value is `len(nodes)` — a count. The
user's text is reduced to `ord(c) % 2` bits and never affects the wording. No model is
called anywhere in the file. This is the same defect as `TRANSCENDPLEXITY`'s
`TransformerBaseline` (two matmuls labelled "Transformer") and its self-named success
metric: **the label says intelligence, the code says lookup table.**

Note the nuance found on the device (`docs/PHONE_BUILD.md` §4): the phone build's router is
a **keyword dispatcher with real side effects** — `battery`/`torch`/`vibrate`/`rsi` genuinely
call hardware and compute. Only the *knowledge* branches are canned. So it is not a failed
model; it is a device-control agent whose question-answering branch is a template.

Also worth noting: `RecursiveSelfImprovementEngine` optimises against **a single hardcoded
3×3 ARC-style pair**:

```python
[{"input": [[0,1,0],[1,1,1],[0,1,0]], "output": [[1,1,1],[1,1,1],[1,1,1]]}]
```

One sample, so "fitness" is not a measurement of anything. Verified live on the device: the
loop returns `Elite Policy: gen0_p0 | Fitness: 0.0` — the elite is the first member of the
initial random population, so the optimisation never improves on generation zero.

## 3. The UI shell is actually good

Credit where due — the generated `index.html` is a clean, responsive Tailwind page. Two
things about it were worth learning from:

- **The API shape.** `POST /api/chat` with `{"messages": [{"role","content"},…]}` and a
  `{"reply": …}` response is the conventional OpenAI-shaped contract. Our first build only
  spoke SSE. That was a real gap, and the generated file is what exposed it.
- **The layout.** Card strip on top, thread below, composer pinned at the bottom.

What was not kept: the Tailwind CDN and Google Fonts (external dependencies where our build
is offline and stdlib-only), and the four fake controls (Battery / Haptic / Torch On /
Torch Off), which have nothing behind them.

## 4. What was built instead

`deploy_node.py` + `deploy_node.html` (this repo) — the same contract, the same visual
layout, a real model behind it.

| | generated original | this port |
|---|---|---|
| reply source | f-string with a node count | `GMGPT` forward pass |
| hardware layer | real, but Termux/device-only | removed (not portable to Windows) |
| RSI layer | 1 hardcoded sample | removed |
| chat API | FastAPI, JSON only | stdlib, **SSE streaming** + JSON |
| runs as generated | **no** (`ModuleNotFoundError`) | yes (11 tests) |
| dependencies | fastapi, uvicorn, pydantic | none beyond the venv |
| reports its own limits | no | yes — params, step, PPL, tok/s |

The transport change also went into `chat_server.py` itself, so there is **one** generation
path. `deploy_node.py` streams the model server's SSE through byte-for-byte rather than
reimplementing generation:

```python
# chat_server.py
if req.get("stream") is False:          # -> {"reply": "...", ...}  (buffered JSON)
    ...same stream_reply() loop, JSON transport...
# default remains text/event-stream (existing front-ends unaffected)
```

```python
# deploy_node.py -- pass tokens through as they are produced
up = open_model_stream(messages, port, timeout, max_tokens)
while chunk := up.read(512):
    self.wfile.write(chunk); self.wfile.flush()
```

### A bug this port found in itself

The first version of `deploy_node.html` parsed **SSE**, but the dashboard answered the
default request with **buffered JSON** — so the page rendered `(empty)` for every prompt.
Caught by driving the real UI in a browser, not by the unit tests. Fixed by making the
dashboard stream by default (`stream: false` opts into JSON), and locked with
`test_deploy_node_streams_sse_by_default`. Verified live: 160 tokens at 45 tok/s through the
page, with the tok/s readout updating. A dashboard that buffers the whole reply defeats the
point of a dashboard.

### Running it

```powershell
python chat_server.py --checkpoint runs/chat_physics_lm/best.pt --port 8765
python deploy_node.py --port 8767 --model-port 8765
# http://127.0.0.1:8767/
```

If the model is not up, the dashboard says so instead of inventing a battery reading:

```
[deploy] WARNING: no model on 8765 -- URLError: ...
[deploy] start it with:  python chat_server.py --checkpoint runs/chat_physics_lm/best.pt --port 8765
```

## 5. Honest note on output quality

The port does not make the model good, and it does not pretend to. At 3.3M parameters the
model emits word-shaped text without coherent answers — measured, live:

```
POST /api/chat  {"messages":[{"role":"user","content":"what is gravity"}],"stream":false}
-> {"reply": "is s The. a Cts a C, is,,  rey to the an hci p the go s Aaret l\ncen can are h an r",
    "tokens": 64, "finish": "length", "model_seconds": 1.38}
```

That is a real forward pass, and it looks worse than the phone build's f-string, which always
produced a fluent sentence. **This is the trade the whole project is about:** a fluent fake
or an honest small model. The dashboard shows params, step, PPL and tok/s precisely so the
answer can be judged rather than admired.

The fix for quality is more parameters and more data, not a better UI — see `RESULTS.md` §4.

## 6. Regression guards

`tests/test_deploy_node.py` (11 tests) locks the audit in:

- the JSON `reply` mode (`stream: false`) returns the documented shape;
- the reply is **not** the f-string (no `"Analysis for"`, no `"spatial topology nodes"`);
- the default path streams **SSE** and the page reads it incrementally (`getReader()`);
- the page does not opt out of streaming;
- **the AST contains no reference to any device-only class or module** (AST, not text, so the
  docstring that documents the removal does not trip its own guard). Note: this guard is
  about *scope*, not existence — those classes are real on the phone (`docs/PHONE_BUILD.md`),
  but they are Termux-only and out of scope for this Windows port;
- `deploy_node.html` contains no fake control;
- `probe_model()` reports offline cleanly instead of raising;
- the dashboard proxies a real reply from the model server (both transports).
