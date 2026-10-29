"""Tests for the chat layer: template, masking split, stop handling, streaming."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chat_server import _hold_partial_marker
from gmllm.chat import (A, E, S, U, messages_to_prompt_with_budget,
                        render_prompt, render_training_example,
                        split_prompt_and_answer, strip_stop)
import torch


def test_render_prompt_shape():
    prompt = render_prompt([
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you?"},
    ])
    assert prompt == f"{S}be nice{E}{U}hi{E}{A}hello{E}{U}how are you?{E}{A}"
    assert prompt.endswith(A)


def test_render_skips_empty_turns():
    prompt = render_prompt([{"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "   "}])
    assert prompt == f"{U}hi{E}{A}"


def test_training_example_masks_only_the_answer():
    full = render_training_example("capital of France?", "Paris.", system="sys")
    assert full == f"{S}sys{E}{U}capital of France?{E}{A}Paris.{E}"
    prompt, answer = split_prompt_and_answer(full)
    assert prompt == f"{S}sys{E}{U}capital of France?{E}{A}"
    assert answer == f"Paris.{E}"
    assert prompt + answer == full


def test_split_with_multiple_turns_uses_last_assistant():
    full = f"{U}a{E}{A}b{E}{U}c{E}{A}d{E}"
    prompt, answer = split_prompt_and_answer(full)
    assert prompt.endswith(f"{U}c{E}{A}")
    assert answer == f"d{E}"


def test_strip_stop_cuts_at_first_marker():
    assert strip_stop("hello<|end|>ignored") == "hello"
    assert strip_stop("hello<|user|>next") == "hello"
    assert strip_stop("plain reply") == "plain reply"


def test_budget_drops_oldest_turns_but_keeps_latest():
    msgs = [{"role": "user", "content": "old " * 50},
            {"role": "assistant", "content": "older " * 50},
            {"role": "user", "content": "newest question"}]
    prompt = messages_to_prompt_with_budget(
        msgs, lambda s: [0] * len(s), max_tokens=80, system="sys")
    assert "newest question" in prompt
    assert "old old" not in prompt
    assert S in prompt and "sys" in prompt
    assert prompt.endswith(A)


def test_partial_marker_is_held_back():
    assert _hold_partial_marker("answer <|en") == "answer "
    assert _hold_partial_marker("answer <|") == "answer "
    assert _hold_partial_marker("answer <") == "answer "
    assert _hold_partial_marker("answer <|end|>") == "answer <|end|>"  # full marker resolved
    assert _hold_partial_marker("2 < 3") == "2 < 3"
    assert _hold_partial_marker("plain text") == "plain text"


# ------------------------------------------------------- atomic control markers
def test_specials_are_single_tokens():
    """The fix for the unreachable-stop-marker ceiling: markers are atomic."""
    from gmllm.chat import MARKERS
    from gmllm.tokenizer import ByteBPETokenizer

    tok = ByteBPETokenizer.train(["the quick brown fox"] * 30, vocab_size=300)
    assert tok.specials == [] and tok.id_for_special(E) == -1
    ids = tok.add_special_tokens(list(MARKERS))
    assert ids == sorted(ids) and len(set(ids)) == 4
    for m in MARKERS:
        assert tok.encode(m) == [tok.id_for_special(m)], m
        assert tok.decode(tok.encode(m)) == m


def test_specials_stay_lossless_when_glued_to_text():
    from gmllm.tokenizer import ByteBPETokenizer

    tok = ByteBPETokenizer.train(["the quick brown fox"] * 30, vocab_size=300)
    tok.add_special_tokens([U, A, S, E])
    for text in ["a<|end|>b", "<|user|>hi<|end|>", "x<|assistant|>y", "  <|end|>  ",
                 "plain words only", "<|end|>"]:
        assert tok.decode(tok.encode(text)) == text, text


def test_special_ids_survive_save_and_load(tmp_path):
    from gmllm.chat import MARKERS
    from gmllm.tokenizer import ByteBPETokenizer

    tok = ByteBPETokenizer.train(["the quick brown fox"] * 30, vocab_size=300)
    tok.add_special_tokens(list(MARKERS))
    before = {m: tok.id_for_special(m) for m in MARKERS}
    p = tmp_path / "tok.json"
    tok.save(p)
    reloaded = ByteBPETokenizer.load(p)
    assert reloaded.vocab_size == tok.vocab_size
    assert {m: reloaded.id_for_special(m) for m in MARKERS} == before


def test_resize_vocab_preserves_learned_rows():
    """Growing the vocab for new markers must not disturb the old embeddings."""
    from gmllm.model import GMConfig, GMGPT

    cfg = GMConfig(vocab_size=300, d_model=32, n_head=4, n_layer=1, ctx=32)
    model = GMGPT(cfg)
    old_emb = model.tok_emb.weight.detach().clone()
    model.resize_vocab(304)
    assert model.cfg.vocab_size == 304
    assert model.tok_emb.weight.shape == (304, 32)
    assert torch.equal(model.tok_emb.weight[:300].detach(), old_emb)
    assert model.lm_head.weight is model.tok_emb.weight        # still tied
    # the resized model must run and produce the wider logits
    logits, _, _, _ = model(torch.tensor([[1, 2, 3]]))
    assert logits.shape[-1] == 304

