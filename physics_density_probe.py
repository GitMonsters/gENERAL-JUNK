#!/usr/bin/env python3
"""Probe physics-page density in a bounded prefix of a large wiki dump.

Simplewiki's full ~250k articles only contain ~700 physics pages (0.27%).  To
scale the corpus we need a bigger source, so this streams a *bounded byte
prefix* of the English Wikipedia dump and reports how many physics pages per MB
it finds -- the number that decides whether enwiki is worth the bandwidth.

    python physics_density_probe.py --url <dump> --cap-mb 150 --label enwiki
"""

from __future__ import annotations

import argparse
import bz2
import io
import json
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_physics_corpus import MAX_CHARS, MIN_CHARS, UA, clean, is_physics

T = Path.home() / "AppData/Local/Temp/opencode"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--cap-mb", type=float, default=150.0)
    ap.add_argument("--label", default="wiki")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else T / f"physics_probe_{args.label}.jsonl"
    cache = T / "dumps" / f"_probe_{args.label}.bz2"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cap = int(args.cap_mb * 1e6)

    if not (cache.exists() and cache.stat().st_size > 1e6):
        req = urllib.request.Request(
            args.url, headers={"User-Agent": UA, "Range": f"bytes=0-{cap-1}"})
        t0, got = time.time(), 0
        with urllib.request.urlopen(req, timeout=180) as r, open(cache, "wb") as f:
            while got < cap:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
        print(f"[dl] {args.label}: {got/1e6:.1f} MB in {(time.time()-t0)/60:.1f} min",
              flush=True)
    else:
        print(f"[dl] {args.label}: cached {cache.stat().st_size/1e6:.1f} MB", flush=True)

    kept = scanned = chars = 0
    samples = []
    with bz2.open(cache, "rb") as fh, open(out, "w", encoding="utf-8") as fps:
        title = body = None
        try:
            for _, el in ET.iterparse(io.BufferedReader(fh), events=("end",)):
                tag = el.tag.rsplit("}", 1)[-1]
                if tag == "title":
                    title = el.text or ""
                elif tag == "text":
                    body = el.text or ""
                elif tag == "page":
                    scanned += 1
                    if title and body:
                        bc = clean(body)
                        if len(bc) >= MIN_CHARS and is_physics(title, bc):
                            fps.write(json.dumps({"title": title.strip(),
                                                  "source": args.label,
                                                  "text": bc[:MAX_CHARS]},
                                                 ensure_ascii=False) + "\n")
                            kept += 1
                            chars += min(len(bc), MAX_CHARS)
                            if len(samples) < 8:
                                samples.append(title.strip())
                    el.clear()
                    title = body = None
        except EOFError:
            pass
    mb = cache.stat().st_size / 1e6
    print(f"[{args.label}] scanned {scanned} pages -> {kept} physics "
          f"({chars/1e6:.2f} M chars) from {mb:.0f} MB  |  "
          f"density {kept/max(scanned,1)*100:.2f}%  -> {out}", flush=True)
    print(f"   sample titles: {samples}", flush=True)


if __name__ == "__main__":
    main()
