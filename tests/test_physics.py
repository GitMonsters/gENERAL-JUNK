"""Physics-corpus construction and continued-pretrain (--init) behaviour."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from build_physics_corpus import clean, is_physics
from gmllm.model import GMConfig, GMGPT

PHYS_PAGE = (
    "In physics, a force is any interaction that changes the motion of an "
    "object. Newton's second law says force equals mass times acceleration. "
    "The SI unit of force is the newton. Friction and gravity are forces. "
    "Pressure is force per unit area, and momentum and energy are conserved. "
    "The acceleration of a body is proportional to the force on it."
)
MILITARY_PAGE = (
    "The armed force of the Gallian military suffered heavy manpower losses "
    "during the war. The squad's mechanics were praised by reviewers of this "
    "video game, whose story follows a fictional empire and its battles."
)


def test_clean_strips_wikitext_markup():
    out = clean("{{Infobox|x=1}}<ref>cite</ref>[[Physics|physics]] ''is'' [[fun]]")
    for junk in ("{{", "}}", "<ref", "[[", "''"):
        assert junk not in out
    assert "physics" in out and "fun" in out


def test_is_physics_accepts_physics_rejects_military_prose():
    assert is_physics("Force", PHYS_PAGE) is True
    # 'force'/'mechanics' appear, but as military/game prose -> not physics
    assert is_physics("Valkyria Chronicles III", MILITARY_PAGE) is False


def test_is_physics_rejects_project_namespaces():
    assert is_physics("Wikipedia:BASIC English wordlist",
                      PHYS_PAGE + " atom electron " * 5) is False
    assert is_physics("Category:Physics", PHYS_PAGE + " mass energy " * 5) is False


def test_build_physics_corpus_reads_partial_dump(tmp_path):
    """A truncated bz2 dump must still yield the pages before the cut."""
    import bz2

    import build_physics_corpus as bpc

    xml = ("<mediawiki>"
           "<page><title>Force</title><revision><text>"
           + PHYS_PAGE * 5 +
           "</text></revision></page>"
           "<page><title>Valkyria Chronicles III</title><revision><text>"
           + MILITARY_PAGE * 5 +
           "</text></revision></page>"
           "</mediawiki>")
    dump = tmp_path / "mini.xml.bz2"
    dump.write_bytes(bz2.compress(xml.encode("utf-8")))

    out = tmp_path / "corpus.jsonl"
    with open(out, "w", encoding="utf-8") as fps:
        kept = bpc.extract(dump, "test", fps)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert kept == 1 and len(rows) == 1
    assert rows[0]["title"] == "Force" and rows[0]["source"] == "test"


def test_init_checkpoint_continues_training(tmp_path):
    """train.py --init must load weights (not reinitialise) and keep training.

    We exercise the mechanism directly rather than shelling out: build a tiny
    checkpoint, load it the way train.py does, step the optimiser, and confirm
    (a) all rows are preserved on load and (b) the weights actually move.
    """
    cfg = GMConfig(vocab_size=64, d_model=32, n_head=4, n_layer=1, ctx=32)
    src = GMGPT(cfg)
    ckpt = tmp_path / "src.pt"
    torch.save({"model": src.state_dict(), "config": cfg.to_dict(),
                "step": 7, "val_ppl": 12.3, "tokenizer": str(tmp_path / "t.json")},
               ckpt)

    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg2 = GMConfig.from_dict(loaded["config"])
    model = GMGPT(cfg2)
    model.load_state_dict(loaded["model"])
    assert loaded["step"] == 7
    for k, v in src.state_dict().items():
        assert torch.equal(model.state_dict()[k], v), k

    before = model.tok_emb.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (2, 32))
    logits, _, loss, _ = model(x, targets=x)
    opt.zero_grad(); loss.backward(); opt.step()
    assert not torch.equal(before, model.tok_emb.weight.detach()), \
        "weights did not move: --init would be a no-op"


# --------------------------------------------- SFT physics-preservation (--lm-data)
def test_load_lm_chunks_packs_fixed_length(tmp_path):
    """The auxiliary LM corpus must load into exact-length next-token chunks."""
    from gmllm.tokenizer import ByteBPETokenizer
    from sft import load_lm_chunks

    tok = ByteBPETokenizer.train(["physics is the study of matter and energy. " * 40],
                                 vocab_size=300)
    path = tmp_path / "prose.jsonl"
    path.write_text(
        "\n".join(json.dumps({"text": "force equals mass times acceleration. " * 5})
                  for _ in range(10)),
        encoding="utf-8")
    chunks = load_lm_chunks(str(path), tok, ctx=64)
    assert chunks.dim() == 2 and chunks.shape[1] == 64 and len(chunks) > 0
    assert chunks.max().item() < tok.vocab_size


def test_lm_aux_loss_is_a_real_gradient_term(tmp_path):
    """Adding the LM term must change the total loss (it is not a no-op)."""
    from sft import lm_loss, masked_loss

    cfg = GMConfig(vocab_size=64, d_model=32, n_head=4, n_layer=1, ctx=32)
    model = GMGPT(cfg)
    idx = torch.randint(0, 64, (2, 16))
    labels = idx.clone()
    chat = masked_loss(model, idx, labels)
    chunks = torch.randint(0, 64, (2, 32))
    lm = lm_loss(model, chunks)
    assert torch.isfinite(chat) and torch.isfinite(lm)
    assert not torch.equal(chat, lm)
    # and the weighted sum is differentiable w.r.t. the model
    total = chat + 0.5 * lm
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
