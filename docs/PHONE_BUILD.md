# The phone build: `octotetrahedral-agi` on the Pixel 7a

The two modules `octotetrahedral_spatial_rsi.py` and `octotetrahedral_device_assistant.py`
**exist, are implemented, and are running on the user's Pixel 7a.** This document records
what they actually do, verified against the live device, and corrects three earlier claims
that were wrong.

Pulled via `adb pull /sdcard/Download/octo-phone` → `phone_recovered/octo-phone/`.
13 files, 96,260 bytes, including `__pycache__/*.cpython-314.pyc`.

---

## 1. Correction log

Three claims in earlier documents were **wrong**. They are corrected here.

| # | earlier claim | verdict | evidence |
|---|---|---|---|
| 1 | "These modules do not exist anywhere" | **FALSE** | Both are on the device at `~/octotetrahedral-agi/`, with compiled `.pyc` proving they executed |
| 2 | "`PixelHardwareBridge` is a fabricated stub" | **FALSE** | Returns a live battery reading: `Battery: 29% \| Status: CHARGING \| Health: GOOD` |
| 3 | "The reply is a hardcoded f-string with a node count" | **HALF TRUE** | It *is* an f-string with a node count — but it is deliberate intent routing, not a broken model (§4) |

Claim 1 arose from searching only the 43 public repos. Claim 2 and 3 arose from reading a
pasted `server.py` without ever exercising it. **Both were failures of method, not of the
device.** The fix for claim 2 is the general lesson: a bridge is proven by *calling* it, not
by reading it.

What remains true from the earlier audit: neither module is in any of the 43 public
GitMonsters repos. `~/octotetrahedral-agi/` on the device is a **superset** of the public
repo — the public clone is missing four locally generated files.

---

## 2. How the evidence was obtained

`run-as com.termux` fails (`package not debuggable` — Play Store build), so adb cannot read
Termux's private storage. But the phone's own server runs *as* the Termux user, so the
hardware path is reachable by talking to it:

```powershell
adb forward tcp:18000 tcp:8000
# then from the PC:
POST http://127.0.0.1:18000/api/battery
```

The tilde in the `PATH` was the whole problem: earlier attempts ran `curl` inside
`adb shell`, which is the Android `shell` user with no `curl` and no Termux binaries.

Confirmed running on the phone:

```
12135  python3.14 .../bin/uvicorn server:app --host 127.0.0.1 --port 8000
/proc/net/tcp: 0100007F:1F40 LISTEN        # 0x1F40 = 8000 on 127.0.0.1
pm list packages: com.termux, com.termux.api   # Termux:API companion installed
```

---

## 3. `PixelHardwareBridge` — real, not a stub

`octotetrahedral_device_assistant.py:28-63`. Every method shells out to a Termux:API binary
via `subprocess.run(cmd, capture_output=True, text=True, timeout=15)`.

| method | binary | verified live? |
|---|---|---|
| `get_battery()` | `termux-battery-status` | **YES** — parsed JSON → `29% / CHARGING / GOOD` |
| `control_torch(state)` | `termux-torch` | routed (HTTP 200) |
| `vibrate(ms)` | `termux-vibrate -d <ms>` | routed (HTTP 200) |
| `speak(text)` | `termux-tts-speak` | not exercised |
| `listen()` | `termux-speech-to-text` | not exercised |

`get_battery` is the strongest proof because it **parses a JSON response**
(`json.loads`, then `.get('percentage')`). A fabricated stub cannot produce
`Health: GOOD`; that string comes from the Android battery API. The `com.termux.api`
package being installed independently corroborates it.

Note the error handling is real too: on failure `run_cmd` returns
`"Hardware Error: ..."` and `get_battery` falls back to `"Raw Battery Status: ..."`.
It degrades instead of crashing.

---

## 4. The chat reply — intent routing, stated honestly

`server.py:24-56` is a **keyword router**, not a language model. There is no `.pt`, no
tokenizer, no generation anywhere in this build.

```
if   "battery"  in msg  → call the real hardware bridge, report live telemetry
elif "torch"    in msg  → actually toggle the flashlight
elif "vibrate"  in msg  → actually vibrate
elif "rsi"      in msg  → actually run the RSI loop
elif "agi"      in msg  → a canned definition paragraph
elif "what is" / "define" / "explain" in msg
                        → "Evaluated concept across N spatial topology nodes"
else                    → "[msg] — Concept mapped onto OctoTetrahedral node array (N vertices)"
```

So the earlier reading was **correct about the mechanism**: the fallback *is* an f-string
that embeds a computed node count. What it got wrong was calling this a defect. It is
consistent with how the rest of the build behaves — the router performs **real side effects**
(battery, torch, vibrate, RSI) for four of its branches, and only the two *knowledge*
branches are canned. The honest description is:

> This is a **device-control dashboard with a keyword router**, not an LLM. It reliably
> controls hardware; it does not answer questions from knowledge.

The grid in the fallback is decorative, but not random in the way it looks:

```python
grid = [[ord(c) % 2 for c in (latest_msg + "123456789")[i:i+3]] for i in range(0, 9, 3)]
```

It hashes the message into a 3×3 binary grid, then runs the *real* `SpatialTopologyEngine`
on it. The computation is genuine; only the interpretation ("stable structural convergence")
is decorative.

---

## 5. `SpatialTopologyEngine` — real connected-component analysis

`octotetrahedral_spatial_rsi.py:177-241`. This is a legitimate spatial-analysis module.

With NumPy + SciPy it runs true connected-component labelling
(`scipy.ndimage.label`) over each colour with a 4- or 8-connected structuring element, and
computes per-component `RSIMetadata`: area, absolute centroid, **normalised** centroid,
normalised bbox, aspect ratio, solidity (`area / bbox_area`), and boundary flags. That is a
normal ARC-style featuriser, competently written.

**One real bug.** When NumPy is absent, the fallback at lines 223-239 does not do
connected-component analysis at all — it returns **one node covering the entire grid**,
with hardcoded `rsi_centroid=(0.5, 0.5)`, `solidity=1.0`, and all four boundary flags `True`.
It ignores the actual grid contents entirely.

This matters because the module's docstring claims it is "Designed for … Termux Python 3.14"
and the import guard exists precisely to survive Termux. On the device the fallback is what
runs, so `extract_spatial_graph` returns `1` for every input. That is why the RSI loop scores
`Fitness: 0.0` (§6) and why the chat router always says "1 active spatial vertex".

The `except (ImportError, Exception)` at line 18 is also a real smell — it swallows *any*
error during import, including a genuine SciPy C-extension crash, and silently degrades.

---

## 6. `RecursiveSelfImprovementEngine` — a real genetic loop, weak fitness

Lines 66-170 are a genuine genetic algorithm: random population init, per-parameter Gaussian
mutation gated by `mutation_rate`, truncation selection (`top max(2, pop//2)`), elitism via
`elite_archive`, and a best-policy return. The mechanics are correct.

The fitness function is where it falls short:

```python
structural_fidelity = min(1.0, len(nodes) / max(1, len(set(str(task["output"])))))
if structural_fidelity >= policy.solidity_threshold:
    success_count += 1
```

It never compares against the task's expected output — `set(str(output))` counts the distinct
*characters* in the output grid's string form, and `nodes` comes from the engine above. It is
a proxy for "did we segment into roughly enough pieces", not task accuracy. Combined with the
single-node fallback (§5), `len(nodes)` is always `1`.

**Live result confirms this** — the improvement loop does not improve:

```
POST /api/rsi → Elite Policy: gen0_p0 | Fitness: 0.0 | Connectivity: 4
```

`gen0_p0` is the *first member of the initial random population*, and fitness is `0.0`. Across
generations the elite never beats generation zero, because the fitness signal is saturated:
with `solidity_threshold` in `[0.2, 0.8]` and `structural_fidelity` pinned at `1/2 = 0.5`
from the single-node fallback, most policies tie, and `sort()` is stable, so the original
first element wins.

So: **a correctly implemented genetic loop driving a fitness function that cannot
discriminate.** The machinery is real; the objective is not.

---

## 7. `update_brain.py` — a code rewriter, not learning

`update_brain.py` does **not** train anything. It is a self-modifying script: it regex-matches
the `api_chat` function in `server.py` and rewrites it with a hardcoded replacement string.
The replacement body is byte-for-byte identical to the `api_chat` already in `server.py`.

So running it is a no-op in practice: it searches for the pattern, replaces it with the same
logic, and prints `[OK] Upgraded server.py with intelligent chat responses!`. The name
"update_brain" is aspirational; the operation is a text substitution. There is no model, no
weights, and no persistable state anywhere in the build.

Also note `make_files.py`, `make_server.py`, `make_server_file.py` exist in three slightly
different sizes (9018 / 13110 / 3491 bytes) — the generator was iterated on-device, and
`~/make_server.py` (15,863 B) is a *fourth*, larger copy outside the repo directory.

---

## 8. What this build actually is

Honest summary:

| layer | status |
|---|---|
| Termux:API hardware bridge (battery/torch/vibrate/tts/stt) | **real and working** |
| Spatial connected-component analysis (NumPy/SciPy path) | **real, competently written** |
| Spatial analysis (pure-Python fallback path) | **degenerate** — always 1 node |
| Genetic RSI loop mechanics | **real** |
| RSI fitness function | **cannot discriminate**; elite never improves on gen 0 |
| Keyword intent router with real side effects | **real** |
| Language model | **absent** — canned strings, no weights |
| Voice I/O | present, via Termux API (not exercised here) |

It is a **working Android device-control agent with genuine spatial-analysis and
meta-optimisation components**, plus a keyword router standing in for a language model.
That is a much more substantial artifact than "fabricated dashboard", and it is genuinely
not in any public repo.

## 9. What this does *not* change

- The **upstream `octotetrahedral-agi` LM still has no causal mask** (`REPO_PICK.md` §2,
  `RESULTS.md` §2b). That finding is independent of the phone build and unaffected.
- The **port in this repo** (`deploy_node.py`, `chat_server.py`) is still the stdlib-only,
  real-model path. The phone build has no model; the port has one but no hardware. They are
  complementary, and neither is a superset of the other.
- `photoncore-x-design/prototype/hardware_abstraction.py` remains the one *properly
  structured* HAL in the ecosystem (typed backends, ABC), but it targets photonics and is
  unrelated to this device layer.

## 10. Reproducing the verification

```powershell
$adb = "<platform-tools>\adb.exe"
& $adb forward tcp:18000 tcp:8000
Invoke-WebRequest http://127.0.0.1:18000/api/battery -Method POST -UseBasicParsing
# → {"result":"Battery: NN% | Status: ... | Health: ...","message":"Battery status pulled successfully."}
& $adb forward --remove tcp:18000
```

Caveat: `adb forward` opens the phone's dashboard to this PC while active. Remove the
forward when finished.
