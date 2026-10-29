#!/usr/bin/env python3
"""Build a LARGE physics corpus by streaming the whole simplewiki dump.

The 60 MB prefix gave 378 pages / 2 MB.  For scaling we want 10-50x: stream the
full dump (358 MB) and keep every physics page.  Memory stays bounded (iterparse);
we stop early if the target page count is reached, so this is also usable as a
"download as much as needed" tool.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
import bz2
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_physics_corpus import MAX_CHARS, MIN_CHARS, UA, clean, is_physics

URL_SIMPLEWIKI = ("https://dumps.wikimedia.org/simplewiki/latest/"
                  "simplewiki-latest-pages-articles.xml.bz2")
URL_ENWIKI = ("https://dumps.wikimedia.org/enwiki/latest/"
              "enwiki-latest-pages-articles.xml.bz2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=URL_SIMPLEWIKI,
                    help="dump URL to stream (default simplewiki; enwiki is much "
                         "denser in physics pages -- ~13%% vs ~0.3%%)")
    ap.add_argument("--source-label", default=None,
                    help="value written to the 'source' field (default: derived from URL)")
    ap.add_argument("--target-pages", type=int, default=20000)
    ap.add_argument("--cap-mb", type=float, default=400.0)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    source_label = args.source_label or (
        "English Wikipedia" if "enwiki" in args.url else
        "Simple English Wikipedia" if "simplewiki" in args.url else "wiki")
    cache = Path(args.cache) if args.cache else (
        Path.home() / "AppData/Local/Temp/opencode/dumps" /
        (Path(args.url).name.replace("latest-", "")))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cap = int(args.cap_mb * 1e6)

    if cache.exists() and cache.stat().st_size > 1e6:
        print(f"[dl] cached {cache.name} ({cache.stat().st_size/1e6:.1f} MB)", flush=True)
    else:
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
                if got % (20 << 20) < (1 << 20):
                    print(f"     {got/1e6:6.1f} MB  ({got/1e6/max(time.time()-t0,1):.2f} MB/s)",
                          flush=True)
        print(f"[dl] {got/1e6:.1f} MB in {(time.time()-t0)/60:.1f} min", flush=True)

    kept = scanned = 0
    t0 = time.time()
    with bz2.open(cache, "rb") as fh, open(args.out, "w", encoding="utf-8") as fps:
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
                                                  "source": source_label,
                                                  "text": bc[:MAX_CHARS]},
                                                 ensure_ascii=False) + "\n")
                            kept += 1
                            if kept % 2000 == 0:
                                print(f"     kept {kept} (scanned {scanned})", flush=True)
                            if kept >= args.target_pages:
                                el.clear()
                                break
                    el.clear()
                    title = body = None
        except EOFError:
            print(f"[ex] end of partial dump (scanned {scanned})", flush=True)
    size = Path(args.out).stat().st_size / 1e6
    print(f"[done] {kept} pages -> {args.out} ({size:.1f} MB) in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
