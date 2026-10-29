# Where did `octotetrahedral_spatial_rsi` / `octotetrahedral_device_assistant` come from?

**CORRECTION (final).** These modules **exist** — they are on the user's phone, in a local
clone at `~/octotetrahedral-agi/`, alongside compiled bytecode:

```
~/octotetrahedral-agi/octotetrahedral_spatial_rsi.py
~/octotetrahedral-agi/octotetrahedral_device_assistant.py
~/octotetrahedral-agi/__pycache__/octotetrahedral_spatial_rsi.cpython-314.pyc
~/octotetrahedral-agi/__pycache__/octotetrahedral_device_assistant.cpython-314.pyc
~/octotetrahedral-agi/__pycache__/server.cpython-314.pyc
~/octotetrahedral-agi/test_features.py
~/octotetrahedral-agi/server.py
~/octotetrahedral-agi/make_server.py
~/octotetrahedral-agi/make_server_file.py
~/octotetrahedral-agi/update_brain.py
```

An earlier version of this document said they "do not exist anywhere". **That was wrong.**
The error was conflating two different claims:

| claim | status |
|---|---|
| "not present in any of the 43 public GitMonsters repos" | **verified true** (§1–2, re-done against the full account) |
| "does not exist anywhere" | **false** — they exist on the device, outside the public repos |

The port in this repo hit `ModuleNotFoundError` because it was built against the *public
repo clone*, which does not contain these two files. The phone's `~/octotetrahedral-agi/`
is a **superset** of the public repo: it includes locally generated modules that were never
pushed.

---

## 1. Method (complete, not sampled)

| step | coverage |
|---|---|
| GitHub API repo list | **43 repos** (the first audit said 41; the account has since grown) |
| `git/trees?recursive=1` per repo | **all 43** — every path in every repo |
| filename match `spatial_rsi|device_assistant` | **0 hits** |
| device-side `grep -rln` over `~/` and `/sdcard/` | **10 hits** — including both modules |

Two repos could not be listed (`BEAT-IT`, `twistedWorm-Templat` → HTTP 409, empty
repositories). Neither name suggests a Python package, and both are in the "marketing /
per-task solver" class from `ANALYSIS.md`.

## 2. The repos whose names were the best candidates

Searched specifically because their names suggested device or spatial code:

| repo | files | `.py` | `spatial_rsi`/`device_assistant` |
|---|---|---|---|
| `octocomputer` | 85 | 55 | none |
| `octo-agi` | 2 | 0 | none |
| `OCTO-AGI-DEMO` | 2 | 0 | none |
| `octotetrahedral_agi` | 1 | 0 | none |
| `octotetrahedral-agi` | 66 | 30 | none |
| `quantra` | 29 | 5 | none |
| `atlas-ai-pro-v2` | 1 | 0 | none |
| `photoncore-x-design` | 57 | 20 | **`prototype/hardware_abstraction.py`** |

**`photoncore-x-design` is the one real find in the public account.** It contains a genuine
hardware abstraction layer — properly structured, unlike the phone build's simple
`subprocess` wrapper:

```python
class DeviceType(Enum):
    SIMULATION = "simulation"    # Software physics simulation
    FPGA       = "fpga"          # FPGA emulation
    BENCHTOP   = "benchtop"      # Tabletop optics
    ASIC       = "asic"          # Foundry chip

class PhotonCoreBackend(ABC):        # real ABC with @abstractmethod
```

This is a **staged-device HAL for photonic matrix-vector multiplication** — optical
computing, with backends that graduate from simulation to silicon. It is a coherent design:
an abstract interface, one concrete backend per hardware stage, and a uniform
`load_matrix` / `forward` API. That is what a device layer should look like at industrial
scale. The phone build's `PixelHardwareBridge` is far simpler (five `subprocess` calls to
Termux:API binaries) but, unlike this HAL, it was verified to actually work on real hardware.

It is unrelated to an Android phone (it targets `/dev/photoncore0`, `serial_port`, FPGA IPs),
but it is the closest thing in the public account to a real "hardware bridge".

## 3. What this means for the audit

**Superseded — see `docs/PHONE_BUILD.md`.** The public-repo part of this section is correct;
the conclusion drawn from it was not. The corrected position:

- The module names, class names and method names are **not invented**. They exist on the
  device, in `~/octotetrahedral-agi/`, and they run.
- The `ModuleNotFoundError` in this repo's port was caused by building against the **public
  clone**, which is missing four locally generated files.
- `PixelHardwareBridge` was **verified working** on the Pixel: `POST /api/battery` returns a
  live reading (`Battery: 29% | Status: CHARGING | Health: GOOD`).

What survives: neither module is in any of the 43 public repos, and upstream `octo_serve.py`
(66 files, 30 `.py`) exposes `/`, `/health`, `/tag`, `/analyze`, `/modules` with no hardware,
battery or Termux code. So the *public* repo genuinely lacks this layer — the device has it.


## 4. The device-side files (what we actually recovered)

`grep -rln` over `~/` and `/sdcard/` on the phone returned:

```
/data/data/com.termux/files/home/octotetrahedral-agi/octotetrahedral_spatial_rsi.py
/data/data/com.termux/files/home/octotetrahedral-agi/octotetrahedral_device_assistant.py
/data/data/com.termux/files/home/octotetrahedral-agi/__pycache__/octotetrahedral_spatial_rsi.cpython-314.pyc
/data/data/com.termux/files/home/octotetrahedral-agi/__pycache__/octotetrahedral_device_assistant.cpython-314.pyc
/data/data/com.termux/files/home/octotetrahedral-agi/__pycache__/server.cpython-314.pyc
/data/data/com.termux/files/home/octotetrahedral-agi/test_features.py
/data/data/com.termux/files/home/octotetrahedral-agi/server.py
/data/data/com.termux/files/home/octotetrahedral-agi/make_server.py
/data/data/com.termux/files/home/octotetrahedral-agi/make_server_file.py
/data/data/com.termux/files/home/octotetrahedral-agi/update_brain.py
```

Evidence they have genuinely executed: **compiled `.pyc` for both modules and for
`server.py`** (`cpython-314`). A module that never imported cannot leave bytecode.

Note also `~/make_server.py` (15.8 KB) at the home root, separate from the copy inside
`octotetrahedral-agi/` — so the generator exists in two places.

## 5. Why the local copy was missed

The port in this repo was built by cloning the **public** repo. `~/octotetrahedral-agi/`
on the device is a **superset**: it contains the public files *plus* locally generated ones
(`octotetrahedral_spatial_rsi.py`, `octotetrahedral_device_assistant.py`,
`update_brain.py`, `test_features.py`) that were never committed. Any audit based on
GitHub alone will miss them, which is exactly what happened here.

The lesson, recorded because it cost real time: **absence from a remote repository is not
absence.** When the artifact under audit runs on a device, the device is the source of truth.

## 6. Resolved — both claims settled by the recovered files

Both open questions are now **closed**, by reading the files *and* exercising the live device.
Full detail in `docs/PHONE_BUILD.md`.

1. **Is the reply an f-string?** — **Yes, but it is intent routing, not a failed model.**
   `update_brain.py` is not learning; it is a regex that rewrites `api_chat` in `server.py`
   with a hardcoded body (byte-for-byte identical to what is already there, so it is a no-op).
   The router performs **real side effects** for `battery` / `torch` / `vibrate` / `rsi`, and
   returns canned strings for the two *knowledge* branches. Description: a device-control
   dashboard with a keyword router, not an LLM. There is no model file anywhere in the build.

2. **Is `PixelHardwareBridge` functional or a stub?** — **Functional.** Verified live via
   `adb forward tcp:18000 tcp:8000`, then `POST /api/battery`:
   ```
   {"result":"Battery: 29% | Status: CHARGING | Health: GOOD",
    "message":"Battery status pulled successfully."}
   ```
   That is a parsed JSON response from `termux-battery-status`, through the Android battery
   API, with the `com.termux.api` package confirmed installed. `Health: GOOD` cannot be
   fabricated by a stub.

Two genuine defects were found in the recovered code, recorded here for accuracy rather than
as a verdict on the whole build:

- **`SpatialTopologyEngine`'s pure-Python fallback is degenerate** (`octotetrahedral_spatial_rsi.py:223-239`).
  With NumPy/SciPy absent it returns **one node covering the entire grid** with hardcoded
  metrics, ignoring the grid contents. Since this is the path that runs on Termux, chat always
  reports "1 active spatial vertex".
- **`RecursiveSelfImprovementEngine`'s fitness cannot discriminate.** It never compares
  against the expected output; it counts distinct characters in the output's string form.
  Live result: `Elite Policy: gen0_p0 | Fitness: 0.0` — the elite is the *first member of the
  initial random population*, and never improves across generations.

