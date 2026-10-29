#!/usr/bin/env python3
"""Build a physics corpus from a *bounded prefix* of the simplewiki dump.

The full 358 MB dump downloads at ~0.3 MB/s here (~20 min before any training).
Physics pages are common enough that a prefix yields hundreds of pages, so we
download only the first N MB, then run the *already-tested* streaming extractor
(build_physics_corpus.extract) over that partial file.

A bz2 prefix is a valid bz2 stream up to the last complete block: decompression
simply stops at the truncation point, which is exactly what we want.

    python build_physics_corpus_fast.py --cap-mb 60
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_physics_corpus import OUT, UA, extract

URL = ("https://dumps.wikimedia.org/simplewiki/latest/"
       "simplewiki-latest-pages-articles.xml.bz2")


def download_prefix(url: str, dest: Path, cap_bytes: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= cap_bytes * 0.98:
        print(f"[dl] cached partial {dest.name} "
              f"({dest.stat().st_size/1e6:.1f} MB)", flush=True)
        return dest
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Range": f"bytes=0-{cap_bytes-1}"})
    t0, got = time.time(), 0
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while got < cap_bytes:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got % (10 << 20) < (1 << 20):
                print(f"     {got/1e6:6.1f} MB / {cap_bytes/1e6:.0f} MB  "
                      f"({got/1e6/max(time.time()-t0,1e-6):.2f} MB/s)", flush=True)
    print(f"[dl] prefix done {got/1e6:.1f} MB in {(time.time()-t0)/60:.1f} min",
          flush=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-mb", type=float, default=60.0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    dest = Path.home() / "AppData/Local/Temp/opencode/dumps/_simplewiki_prefix.bz2"
    p = download_prefix(URL, dest, int(args.cap_mb * 1e6))
    n = 0
    with open(args.out, "w", encoding="utf-8") as fps:
        n = extract(p, "Simple English Wikipedia", fps)
    size = Path(args.out).stat().st_size / 1e6
    print(f"[done] {n} physics pages -> {args.out} ({size:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
