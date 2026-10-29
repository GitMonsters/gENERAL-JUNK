"""Build a physics instruction set from the physics corpus (extractive QA style).

For each physics page: question = a natural prompt built from the title,
answer = the page's lead paragraph (first ~2 sentences).  This mirrors the
upstream repo's own instruction format (extractive QA over their documents) and
gives the SFT stage physics-shaped supervision instead of only Dolly chit-chat.

Writes a JSONL of {"instruction", "response", "context"} rows.
"""
import argparse
import json
import re
from pathlib import Path

TMP = Path.home() / "AppData/Local/Temp/opencode"

ap = argparse.ArgumentParser(description="Extractive physics QA from a corpus")
ap.add_argument("--src", default=str(TMP / "physics_corpus.jsonl"))
ap.add_argument("--out", default=str(TMP / "physics_qa.jsonl"))
ARGS = ap.parse_args()
SRC = Path(ARGS.src)
OUT = Path(ARGS.out)

QMARKS = {
    "what": "What is {t}?",
    "how": "How does {t} work?",
    "why": "Why is {t} important?",
}

rows = []
FRAGMENT = re.compile(r"(alt=|^\s*[a-z:,;|])")
for line in SRC.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    o = json.loads(line)
    text = re.sub(r"\s+", " ", o["text"]).strip()
    title = o["title"]
    sents = re.split(r"(?<=[.!?])\s+", text)
    # drop caption/fragment sentences at the head of the page
    good = []
    started = False
    for s in sents:
        s = s.strip()
        if not started:
            if len(s.split()) < 4 or FRAGMENT.search(s) or not s[:1].isupper():
                continue
            started = True
        good.append(s)
    if not good:
        continue
    lead = ""
    for s in good:
        lead = (lead + " " + s).strip()
        if len(lead) >= 160:
            break
    if len(lead) < 60:
        continue
    kind = ["what", "how", "why"][len(title) % 3]
    q = QMARKS[kind].format(t=title)
    rows.append({"instruction": q, "response": lead[:600],
                 "context": " ".join(good[:4])[:1200]})

OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
               encoding="utf-8")
print(f"{len(rows)} physics QA rows -> {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")
for r in rows[:4]:
    print(f"\nQ: {r['instruction']}\nA: {r['response'][:180]}")
