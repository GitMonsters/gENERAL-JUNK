"""Chat-SFT specifics: data parsing, format learning, marker reachability."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chat_server import token_ban_list
from gmllm.chat import (A, E, S, U, MARKERS, load_qa_pairs, parse_qa_line,
                        read_jsonl, render_prompt)
from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer


# ------------------------------------------------------------------ data parsing
def test_parse_qa_line_legacy_and_v2():
    legacy = parse_qa_line("Instruction: What makes it interesting?\nResponse: A bit.")
    assert legacy == ("What makes it interesting?", "A bit.")
    v2 = parse_qa_line("Question : What can months tell us ?\nResponse : Not much.")
    assert v2 == ("What can months tell us ?", "Not much.")


def test_parse_qa_line_rejects_junk():
    assert parse_qa_line("just a sentence") is None
    assert parse_qa_line("Instruction: no answer\n") is None


def test_read_jsonl_skips_broken_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"text": "a"}\n\nnot json\n{"text": "b"}\n', encoding="utf-8")
    assert [o["text"] for o in read_jsonl(p)] == ["a", "b"]


# ------------------------------------------------------------- learned behaviour
def test_answer_mask_covers_only_the_reply():
    """Training rows must supervise exactly the reply + its end marker."""
    full = render_prompt([{"role": "user", "content": "q"}]).replace(A, "")
    # split_prompt_and_answer is exercised in test_chat.py; here check the
    # invariant that the assistant marker is the final prompt token.
    prompt = render_prompt([{"role": "user", "content": "q"}])
    assert prompt.endswith(A)
    assert prompt.count(A) == 1
    assert full.endswith("q" + E)


def test_token_ban_list_blocks_marker_pieces():
    tok = ByteBPETokenizer.train(
        ["the quick brown fox jumps over the lazy dog"] * 40, vocab_size=300)
    banned = token_ban_list(tok)
    for marker in (E, U, S):
        pieces = tok.encode(marker)
        if len(pieces) > 1:
            assert set(pieces) <= banned


# ------------------------------------------------------ measured model behaviour
CHAT_CKPT = Path(__file__).resolve().parents[1] / "runs" / "chat" / "best.pt"


def _load_chat():
    ckpt = torch.load(CHAT_CKPT, map_location="cpu", weights_only=False)
    cfg = GMConfig.from_dict(ckpt["config"])
    tok = ByteBPETokenizer.load(ckpt["tokenizer"])
    model = GMGPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, ckpt


def test_sft_beat_the_base_model_on_held_out_dolly():
    """The SFT checkpoint must actually be better at replies than its base.

    Base step-99 reports 188.4 *document* PPL; SFT reports answer-token PPL.
    Both are compared here on the identical held-out split, so the comparison
    is apples-to-apples.
    """
    import torch.nn.functional as F

    import eval_chat as ec
    from gmllm.chat import load_dolly_pairs

    base_ck = Path(__file__).resolve().parents[1] / "runs" / "small" / "best.pt"
    dolly_path = Path.home() / "AppData/Local/Temp/opencode/dolly.jsonl"
    if not (CHAT_CKPT.exists() and base_ck.exists() and dolly_path.exists()):
        return  # artefacts not present in this checkout

    pairs = load_dolly_pairs(dolly_path)
    eval_pairs = pairs[200:300]
    scores = {}
    for name, path in (("base", base_ck), ("sft", CHAT_CKPT)):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = GMConfig.from_dict(ckpt["config"])
        tok = ByteBPETokenizer.load(ckpt["tokenizer"])
        model = GMGPT(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval()
        rows = ec.to_rows(eval_pairs, tok, ctx=320, with_context=True)
        ppl, n = ec.answer_ppl(model, rows)
        scores[name] = ppl
    assert scores["sft"] < scores["base"], scores


def test_end_marker_is_unreachable_for_this_checkpoint():
    """Documented limit: raw greedy decoding cannot reach '<|end|>'.

    This test encodes the honest finding rather than a wish: the marker's
    first piece ('<') sits far down the ranking on a chat prompt.  If a future
    retrain fixes it, this test fails loudly and should be updated.
    """
    if not CHAT_CKPT.exists():
        return
    model, tok, ckpt = _load_chat()
    if ckpt.get("step", 0) < 100:
        return
    prompt = render_prompt([{"role": "user", "content": "What is the capital of France?"}])
    with torch.no_grad():
        logits = model(torch.tensor([tok.encode(prompt)]))[0][:, -1, :]
    lt = tok.encode("<")[0]
    rank = int((logits[0] > logits[0, lt]).sum())
    assert rank > 100, f"marker opener rank {rank}: reachability changed, update CHAT.md"


# ------------------------------------------------- atomic-marker server behaviour
ATOMIC_CKPT = Path(__file__).resolve().parents[1] / "runs" / "chat_atomic" / "best.pt"


def test_atomic_checkpoint_uses_stop_ids_and_bans_nothing(tmp_path, monkeypatch):
    """A checkpoint with atomic markers must not ban them — it stops on them.

    This is the server half of the §8 fix: with single-token markers the whole
    "ban the pieces" workaround is wrong, because the marker is now exactly the
    token we *want* the model to be able to predict.
    """
    if not ATOMIC_CKPT.exists():
        return
    import chat_server

    chat_server.load_checkpoint(str(ATOMIC_CKPT))
    assert chat_server.STATE["atomic"] is True
    assert chat_server.STATE["ban"] == set()
    tok = chat_server.STATE["tok"]
    assert chat_server.STATE["stop_ids"] == {tok.id_for_special(m)
                                             for m in (E, U, S)}
    assert chat_server.STATE["meta"]["atomic_markers"] is True


def test_atomic_markers_are_single_ids():
    """The four control markers each encode to exactly one id, and the ids are
    the four consecutive entries appended above the base byte+BPE range."""
    tok = ByteBPETokenizer.train(["the quick brown fox"] * 40, vocab_size=300)
    base_vocab = tok.vocab_size
    ids = tok.add_special_tokens(list(MARKERS))
    assert ids == [base_vocab + i for i in range(len(MARKERS))]
    for m, i in zip(MARKERS, ids):
        assert tok.encode(m) == [i], m

