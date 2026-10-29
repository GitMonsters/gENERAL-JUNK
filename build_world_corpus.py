#!/usr/bin/env python3
"""Build a GENERAL world-knowledge corpus from a wiki dump.

The physics corpus makes a specialist; a chat model also needs broad world
knowledge (people, places, history, science, culture).  This builder keeps any
substantial article rather than filtering to a topic, so the model sees the
encyclopedia as a whole.

Reuses the cached dump from the physics build, so there is no extra download.

    python build_world_corpus.py --out world_corpus.jsonl \
        --cache "%TEMP%/opencode/dumps/_enwiki_prefix.bz2" --target-pages 20000

Only pages that are real articles survive:
  * not a redirect / disambiguation / list page
  * body length in [MIN_CHARS, MAX_CHARS] before truncation
  * lead paragraph exists (so it is prose, not a table dump)
"""

from __future__ import annotations

import argparse
import bz2
import io
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_physics_corpus import MAX_CHARS, MIN_CHARS, UA, clean

T = Path.home() / "AppData/Local/Temp/opencode"

#: Titles that are structurally not articles.
SKIP_TITLE = re.compile(
    r"^(List of|Lists of|Index of|Outline of|Timeline of|Glossary of|"
    r"Category:|Template:|Wikipedia:|Portal:|File:|Help:|Draft:|"
    r"\d{1,4}(st|nd|rd|th)? century|"
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December) \d{1,2}|"
    r"\d{3,4})$",
    re.IGNORECASE,
)
#: Bodies that are mostly tables/lists/infoboxes rather than prose.
LISTISH = re.compile(r"^\s*(\||!|\{\||\*|#)")


def is_article(title: str, body: str) -> bool:
    t = title.strip()
    if len(t) < 3 or not t[0].isalnum():
        return False
    if not t or SKIP_TITLE.match(t):
        return False
    if "disambiguation" in body[:400].lower():
        return False
    # count prose characters in the first ~2000 chars: list pages are low-prose
    head = body[:2000]
    prose = sum(1 for ch in head if ch.isalpha() or ch.isspace())
    if prose < 0.85 * max(len(head), 1):
        return False
    # require at least two real sentences in the lead
    sents = [s for s in re.split(r"(?<=[.!?])\s+", body[:600]) if len(s.split()) >= 5]
    return len(sents) >= 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=(
        "https://dumps.wikimedia.org/enwiki/latest/"
        "enwiki-latest-pages-articles.xml.bz2"))
    ap.add_argument("--cache", default=str(T / "dumps/_enwiki_prefix.bz2"))
    ap.add_argument("--source-label", default="English Wikipedia")
    ap.add_argument("--target-pages", type=int, default=20000)
    ap.add_argument("--cap-mb", type=float, default=700.0)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS)
    args = ap.parse_args()

    cache = Path(args.cache)
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
                if got % (50 << 20) < (1 << 20):
                    print(f"     {got/1e6:6.1f} MB", flush=True)
        print(f"[dl] {got/1e6:.1f} MB in {(time.time()-t0)/60:.1f} min", flush=True)

    kept = scanned = chars = 0
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
                        if (args.min_chars <= len(bc) and
                                is_article(title, bc)):
                            text = bc[: args.max_chars]
                            fps.write(json.dumps(
                                {"title": title.strip(),
                                 "source": args.source_label,
                                 "text": text},
                                ensure_ascii=False) + "\n")
                            kept += 1
                            chars += len(text)
                            if kept % 2000 == 0:
                                print(f"     kept {kept} (scanned {scanned}, "
                                      f"{chars/1e6:.0f} M chars)", flush=True)
                            if kept >= args.target_pages:
                                el.clear()
                                break
                    el.clear()
                    title = body = None
        except EOFError:
            print(f"[ex] end of partial dump (scanned {scanned})", flush=True)

    mb = Path(args.out).stat().st_size / 1e6
    print(f"[done] {kept} pages / {chars/1e6:.1f} M chars -> {args.out} ({mb:.1f} MB) "
          f"in {(time.time()-t0)/60:.1f} min  |  keep-rate "
          f"{kept/max(scanned,1)*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
