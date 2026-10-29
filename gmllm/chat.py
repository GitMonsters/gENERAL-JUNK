"""Chat formatting for GM-GPT.

The base model is a pure next-token LM.  Chat behaviour is added by fine-tuning
(SFT) on conversations rendered with a fixed template, and reproduced exactly at
inference time by this module -- the *same* functions build training rows and
runtime prompts, so there is no train/serve skew.

Template (byte-level BPE, so markers are plain text):

    <|system|>you are a helpful assistant<|end|>
    <|user|>hello<|end|>
    <|assistant|>hi!<|end|>
    <|user|>...<|end|>
    <|assistant|>

The model learns to emit ``<|end|>`` when a reply is complete; the server stops
on any control marker.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

U = "<|user|>"
A = "<|assistant|>"
S = "<|system|>"
E = "<|end|>"

#: All four control markers, in a stable order, for atomic-token registration.
MARKERS = (S, U, A, E)
STOP_MARKERS = (E, U, S)
DEFAULT_SYSTEM = "You are GM-GPT, a small, honest assistant."


def render_training_example(
    instruction: str,
    response: str,
    context: str | None = None,
    system: str | None = None,
) -> str:
    """One training row: optional system, one user turn, one assistant turn."""
    user = instruction.strip()
    if context and context.strip():
        user += "\n\nContext:\n" + context.strip()
    parts = []
    if system:
        parts.append(f"{S}{system}{E}")
    parts.append(f"{U}{user}{E}")
    parts.append(f"{A}{response.strip()}{E}")
    return "".join(parts)


def render_prompt(messages: Sequence[Dict[str, str]]) -> str:
    """Runtime prompt ending with ``<|assistant|>`` ready for generation.

    ``messages`` = [{"role": "system"|"user"|"assistant", "content": ...}, ...]
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            out.append(f"{S}{content}{E}")
        elif role == "assistant":
            out.append(f"{A}{content}{E}")
        else:
            out.append(f"{U}{content}{E}")
    out.append(A)
    return "".join(out)


def split_prompt_and_answer(full: str) -> tuple[str, str]:
    """Split a rendered training row at the last ``<|assistant|>``.

    Returns (prompt_including_marker, answer_including_end_marker).  Used to
    mask the loss so only assistant tokens are supervised.
    """
    idx = full.rfind(A)
    if idx < 0:
        raise ValueError("no assistant marker in example")
    cut = idx + len(A)
    return full[:cut], full[cut:]


def strip_stop(text: str) -> str:
    """Cut generated text at the first control marker."""
    ends = [text.find(m) for m in STOP_MARKERS if m in text]
    if not ends:
        return text
    return text[: min(ends)]


# --------------------------------------------------------------------- data I/O
_QA_RE = re.compile(
    r"^\s*(?:Instruction|Question)\s*:\s*(?P<q>.*?)\s*\n\s*"
    r"(?:Response|Answer)\s*:\s*(?P<a>.*)$",
    re.S,
)


def parse_qa_line(text: str) -> Tuple[str, str] | None:
    """Parse one line of the upstream instruction corpus into (question, answer).

    The GitMonsters instruction files are not request/response pairs in the
    Dolly sense: they are *extractive QA over their own documents* (the answer
    is a verbatim sentence pulled from the same transcript it came from, by
    ``gen_instructions.py``).  Both the legacy ``Instruction:/Response:`` and
    the v2 ``Question :/Response :`` spellings are accepted.
    """
    m = _QA_RE.match(text.strip())
    if not m:
        return None
    q, a = m.group("q").strip(), m.group("a").strip()
    if not q or not a:
        return None
    return q, a


def read_jsonl(path) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file, skipping blank/broken lines."""
    import json
    from pathlib import Path

    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def load_qa_pairs(paths: Iterable) -> List[Tuple[str, str]]:
    """Read one or more upstream instruction JSONL files into (Q, A) pairs."""
    pairs: List[Tuple[str, str]] = []
    for path in paths:
        for obj in read_jsonl(path):
            text = obj.get("text")
            if not isinstance(text, str):
                continue
            parsed = parse_qa_line(text)
            if parsed:
                pairs.append(parsed)
    return pairs


def load_qa_pairs_silent(paths: Iterable) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """Read upstream instruction pairs **without printing anything**.

    Same result as :func:`load_qa_pairs`, but for embedding in tools that must
    not pollute stdout, and -- unlike the plain loader -- it is *accountable*
    about what it discards.  ``read_jsonl`` skips broken JSON and
    ``parse_qa_line`` returns ``None`` on unparseable text; both do so silently,
    so a corpus can lose rows with no trace.  This returns the pairs together
    with a skip ledger::

        pairs, stats = load_qa_pairs_silent(paths)
        # stats -> {"lines": 13769, "nonblank": 13769, "json_objects": 13769,
        #           "bad_json": 0, "unparsed": 0, "missing_text": 0,
        #           "kept": 13769, "files": 2}

    Nothing here writes to stdout/stderr; the caller decides what, if anything,
    to surface.  Verified by ``tests/test_chat_pairs.py`` (captures stdout and
    asserts it stays empty, on both clean and deliberately-corrupted input).
    """
    from pathlib import Path

    pairs: List[Tuple[str, str]] = []
    stats = {"files": 0, "lines": 0, "nonblank": 0, "json_objects": 0,
             "bad_json": 0, "missing_text": 0, "unparsed": 0, "kept": 0}
    for path in paths:
        stats["files"] += 1
        # Read raw text so a malformed line can be *counted* rather than
        # vanishing inside read_jsonl's own silent skip.
        raw = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        for line in raw:
            stats["lines"] += 1
            line = line.strip()
            if not line:
                continue
            stats["nonblank"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            if not isinstance(obj, dict):
                stats["bad_json"] += 1
                continue
            stats["json_objects"] += 1
            text = obj.get("text")
            if not isinstance(text, str):
                stats["missing_text"] += 1
                continue
            parsed = parse_qa_line(text)
            if parsed:
                pairs.append(parsed)
                stats["kept"] += 1
            else:
                stats["unparsed"] += 1
    return pairs, stats


def load_dolly_pairs(path) -> List[Tuple[str, str, str]]:
    """Read Dolly-15k rows into (instruction, response, context)."""
    pairs = []
    for obj in read_jsonl(path):
        ins = (obj.get("instruction") or "").strip()
        resp = (obj.get("response") or "").strip()
        ctx = (obj.get("context") or "").strip()
        if ins and resp:
            pairs.append((ins, resp, ctx))
    return pairs


def messages_to_prompt_with_budget(
    messages: Sequence[Dict[str, str]],
    encode,
    max_tokens: int,
    system: str | None = DEFAULT_SYSTEM,
) -> str:
    """Render a prompt but drop the oldest turns until it fits ``max_tokens``.

    Keeps the system prompt and the most recent messages.  ``encode`` is the
    tokenizer's encode function.
    """
    msgs: List[Dict[str, str]] = list(messages)
    sys_msg = None
    if system:
        sys_msg = {"role": "system", "content": system}
    while True:
        trial = ([sys_msg] if sys_msg else []) + msgs
        prompt = render_prompt(trial)
        if len(encode(prompt)) <= max_tokens or len(msgs) <= 1:
            return prompt
        msgs = msgs[1:]          # drop oldest turn (never the latest)
