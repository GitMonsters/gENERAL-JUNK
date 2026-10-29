#!/usr/bin/env python3
"""Build a physics prose corpus from Wikipedia-family dumps.

Streams the multistream XML dumps (bz2) and keeps only pages that look like
physics, so we never decompress 358 MB to disk.  Two sources:

  * Simple English Wikipedia (simplewiki) -- short, plain sentences; the most
    learnable register for a 3.3M-param model.
  * English Wikibooks (enwikibooks)       -- physics *study guides*, i.e.
    textbook prose rather than encyclopedia stubs.

Filtering is deliberately conservative and explainable: a page is kept if its
title is in the physics title-seed list, OR its body hits >= MIN_HITS distinct
physics terms.  We also drop markup-heavy pages (tables, infoboxes, references)
because a byte-level LM should learn prose, not wikitext syntax.

Output: one JSONL row per page {"title", "source", "text"}.  Deterministic and
re-runnable (same dumps in -> same corpus out).
"""

from __future__ import annotations

import bz2
import io
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = ("gmllm-research/1.0 (local LM training experiment; "
      "contact: local-user) python-urllib")

TMP = Path.home() / "AppData/Local/Temp/opencode"
DUMPS = TMP / "dumps"
OUT = TMP / "physics_corpus.jsonl"

SOURCES = [
    ("simplewiki", "Simple English Wikipedia",
     "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"),
    ("enwikibooks", "English Wikibooks",
     "https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2"),
]

# --- physics relevance -------------------------------------------------------
# Distinct physics terms; a body needs MIN_HITS of them (or a title match).
PHYS_TERMS = [
    "physics", "physical", "quantum", "entropy", "thermodynamic", "velocity",
    "acceleration", "momentum", "force", "energy", "gravity", "gravitational",
    "mass", "electron", "photon", "proton", "neutron", "wavelength",
    "frequency", "relativity", "electromagnetic", "electric field",
    "magnetic field", "friction", "newton", "particle", "atom", "molecule",
    "density", "pressure", "kinetic", "potential energy", "inertia",
    "mechanics", "optics", "magnetism", "radiation", "nuclear", "boson",
    "fermion", "quark", "capacitor", "voltage", "current", "circuit",
    "temperature", "heat", "refraction", "diffraction", "interference",
    "oscillation", "pendulum", "pulley", "lever", "torque", "orbit",
    "escape velocity", "black hole", "supernova", "plasma", "isotope",
    "photoelectric", "doppler", "sound wave", "light", "laser", "semiconductor",
]
TERM_RE = {t: re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in PHYS_TERMS}

TITLE_SEEDS = {
    "physics", "force", "energy", "motion", "gravity", "mass", "velocity",
    "acceleration", "momentum", "atom", "electron", "light", "heat",
    "temperature", "sound", "electricity", "magnetism", "friction", "pressure",
    "density", "wave", "optics", "quantum mechanics", "relativity",
    "thermodynamics", "electromagnetism", "newton's laws of motion",
    "potential energy", "kinetic energy", "special relativity",
    "general relativity", "black hole", "star", "planet", "solar system",
    "physics study guide", "gcse physics", "a-level physics",
}
MIN_HITS = 6
MIN_CHARS = 400          # skip stubs
MAX_CHARS = 12000        # cap very long pages

# strip wikitext scaffolding so we keep prose
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
REF_RE = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")
TABLE_RE = re.compile(r"\{\|.*?\|\}", re.S)
FILE_RE = re.compile(r"\[\[\s*(File|Image|Category)\s*:.*?\]\]", re.S | re.I)
LINK_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
HEAD_RE = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
QUOTE_RE = re.compile(r"''+")
WS_RE = re.compile(r"[ \t]+")


def clean(text: str) -> str:
    text = TABLE_RE.sub(" ", text)
    text = REF_RE.sub(" ", text)
    for _ in range(4):                       # nested templates
        text = TEMPLATE_RE.sub(" ", text)
    text = FILE_RE.sub(" ", text)
    # links: run twice (nested/pipe cases), then strip any residual brackets
    text = LINK_RE.sub(r"\1", text)
    text = LINK_RE.sub(r"\1", text)
    text = text.replace("[[", " ").replace("]]", " ")
    text = HEAD_RE.sub(r"\1.", text)
    text = TAG_RE.sub(" ", text)
    text = QUOTE_RE.sub("", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    text = WS_RE.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def is_physics(title: str, body: str) -> bool:
    t = title.strip()
    low = t.lower()
    # never keep project/meta namespaces or list pages
    if ":" in t and low.split(":", 1)[0] in {
            "wikipedia", "category", "template", "help", "portal", "talk",
            "user", "file", "mediawiki", "draft", "module"}:
        return False
    if low in TITLE_SEEDS:
        return True
    hits = sum(1 for rx in TERM_RE.values() if rx.search(body))
    return hits >= MIN_HITS


def download(name: str, url: str) -> Path:
    DUMPS.mkdir(parents=True, exist_ok=True)
    dest = DUMPS / Path(url).name
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[dl] {name}: cached {dest.name} "
              f"({dest.stat().st_size/1e6:.1f} MB)", flush=True)
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    print(f"[dl] {name}: {url}", flush=True)
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got, t0, last = 0, time.time(), 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got - last > 40 * (1 << 20):
                last = got
                pct = f" {100*got/total:4.0f}%" if total else ""
                print(f"     {got/1e6:6.1f} MB{pct}  "
                      f"({got/1e6/max(time.time()-t0,1e-6):.1f} MB/s)", flush=True)
    print(f"[dl] {name}: done {got/1e6:.1f} MB", flush=True)
    return dest


def extract(path: Path, source: str, fps) -> int:
    """Stream the dump; write physics pages.  Returns pages kept."""
    kept = n = 0
    with bz2.open(path, "rb") as fh:
        raw = io.BufferedReader(fh)
        # iterparse so memory stays bounded even on the 358 MB dump
        title = body = None
        try:
            events = ET.iterparse(raw, events=("end",))
            for event, el in events:
                tag = el.tag.rsplit("}", 1)[-1]
                if tag == "title":
                    title = el.text or ""
                elif tag == "text":
                    body = el.text or ""
                elif tag == "page":
                    n += 1
                    if title and body:
                        body_c = clean(body)
                        if (len(body_c) >= MIN_CHARS and is_physics(title, body_c)):
                            fps.write(json.dumps({
                                "title": title.strip(),
                                "source": source,
                                "text": body_c[:MAX_CHARS],
                            }, ensure_ascii=False) + "\n")
                            kept += 1
                    el.clear()
                if tag == "page":
                    title = body = None
        except EOFError:
            # truncated dump (bounded-prefix download): everything parsed so far
            # is already written; the tail block simply isn't available.
            print(f"[ex] {source}: reached end of partial dump "
                  f"(scanned {n}, kept {kept})", flush=True)
        print(f"[ex] {source}: scanned {n} pages, kept {kept}", flush=True)
    return kept


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(OUT, "w", encoding="utf-8") as fps:
        for name, label, url in SOURCES:
            try:
                p = download(name, url)
                total += extract(p, label, fps)
            except Exception as e:
                print(f"[err] {name}: {type(e).__name__}: {e}", flush=True)
    size = OUT.stat().st_size / 1e6
    print(f"[done] {total} physics pages -> {OUT} ({size:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
