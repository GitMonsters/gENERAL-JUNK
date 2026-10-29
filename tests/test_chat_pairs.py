"""The upstream QA "silent pair" reader: quiet by contract, accountable by design.

Two properties matter and are both pinned here:

1. **Silence** -- reading pairs must print nothing to stdout or stderr.  A tool
   embedded elsewhere cannot be trusted if it writes to the console.
2. **Accountability** -- rows that are discarded (broken JSON, unparseable text,
   missing fields) must be *counted*, because the plain loaders drop them with
   no trace.  Silence must not mean blindness.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmllm.chat import load_qa_pairs, load_qa_pairs_silent

LEGACY = '{"text": "Instruction: What is force?\\nResponse: An interaction that changes motion."}'
V2 = '{"text": "Question : What is energy ?\\nResponse : The capacity to do work."}'


def _write(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "instructions.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_reads_both_upstream_spellings(tmp_path):
    p = _write(tmp_path, [LEGACY, V2])
    pairs, stats = load_qa_pairs_silent([p])
    assert len(pairs) == 2
    assert ("What is force?", "An interaction that changes motion.") in pairs
    assert ("What is energy ?", "The capacity to do work.") in pairs
    assert stats["kept"] == 2


def test_prints_nothing_to_stdout_or_stderr(tmp_path):
    """The whole point: absolute silence, even with junk in the file."""
    p = _write(tmp_path, [
        LEGACY,
        "",                              # blank
        "{not valid json",               # broken JSON
        '{"nope": 1}',                   # valid JSON, no "text"
        '{"text": "no markers here"}',   # text, but unparseable
        V2,
    ])
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        pairs, _ = load_qa_pairs_silent([p])
    assert out.getvalue() == "", f"wrote to stdout: {out.getvalue()!r}"
    assert err.getvalue() == "", f"wrote to stderr: {err.getvalue()!r}"
    assert len(pairs) == 2


def test_skip_ledger_is_exact(tmp_path):
    p = _write(tmp_path, [
        LEGACY,                          # kept
        "",                              # blank (not counted as a loss)
        "{not valid json",               # bad_json
        '{"nope": 1}',                   # missing_text
        '{"text": "no markers here"}',   # unparsed
        V2,                              # kept
    ])
    pairs, stats = load_qa_pairs_silent([p])
    assert stats == {"files": 1, "lines": 6, "nonblank": 5, "json_objects": 4,
                     "bad_json": 1, "missing_text": 1, "unparsed": 1, "kept": 2}
    # every non-blank line is accounted for somewhere
    assert stats["kept"] + stats["bad_json"] + stats["missing_text"] \
        + stats["unparsed"] == stats["nonblank"]
    assert len(pairs) == stats["kept"]


def test_matches_the_plain_loader_on_clean_input(tmp_path):
    """Silent read must agree with the existing loader when nothing is dropped."""
    p = _write(tmp_path, [LEGACY, V2])
    assert load_qa_pairs_silent([p])[0] == load_qa_pairs([p])


def test_bad_json_is_counted_and_does_not_raise(tmp_path):
    p = _write(tmp_path, ["{" * 50, "]", LEGACY])
    pairs, stats = load_qa_pairs_silent([p])
    assert stats["bad_json"] == 2
    assert len(pairs) == 1


def test_missing_file_raises_rather_than_silently_returning_empty(tmp_path):
    """A wrong path is a caller error: fail loudly, do not pretend silence."""
    try:
        load_qa_pairs_silent([tmp_path / "does_not_exist.jsonl"])
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for a missing path")


def test_real_upstream_corpus_is_fully_parsed():
    """On the actual upstream data every row is usable (measured, not assumed)."""
    base = (Path.home() / "AppData/Local/Temp/opencode/gitmonsters/"
            "octotetrahedral-agi/data")
    files = [base / "instructions.jsonl", base / "instructions_v2.jsonl"]
    if not files[0].exists():
        import pytest
        pytest.skip("upstream corpus not downloaded in this environment")
    pairs, stats = load_qa_pairs_silent(files)
    assert stats["bad_json"] == 0 and stats["unparsed"] == 0
    assert stats["missing_text"] == 0
    assert len(pairs) == stats["kept"] == 13769
