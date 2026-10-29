"""Byte-level BPE tokenizer.

Why not the upstream word-level whitespace tokenizer?
  GitMonsters/octotetrahedral-agi/train_transformer.py builds a word vocab with
  min_freq>=2 and maps everything else to <UNK>; their own RESULTS.md documents the
  resulting failures ("trigger words weren't in the vocab and were invisible to the
  model", "char embeddings are a dead feature"). A byte-level base alphabet makes <UNK>
  impossible, and BPE merges recover word-level efficiency.

Pure-Python, dependency-free, trained on whitespace-aggregated word frequencies
(a single pass + ~vocab_size merge passes over the unique-word table, which is fast
enough for corpora up to tens of MB).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


class ByteBPETokenizer:
    """Byte-level BPE. Ids 0..255 are raw bytes; ids >= 256 are learned merges."""

    def __init__(
        self,
        merges: List[Tuple[int, int]] | None = None,
        specials: List[str] | None = None,
    ):
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in (merges or [])]  # type: ignore[misc]
        self.specials: List[str] = list(specials or [])
        self._decode_cache: Dict[int, bytes] = {}
        self._merge_index: Dict[Tuple[int, int], int] = {}
        self._word_cache: Dict[str, List[int]] = {}
        self._rebuild_decode_cache()
        self._rebuild_specials()

    # ------------------------------------------------------------------ train
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 512, min_freq: int = 2,
              specials: List[str] | None = None) -> "ByteBPETokenizer":
        vocab_size = max(256, int(vocab_size))
        freq: Counter = Counter()
        for text in texts:
            freq.update(text.split())

        # Only words that repeat are worth merging against; rare words fall back
        # to raw bytes at encode time (lossless — the base alphabet is bytes).
        words: Dict[str, List[int]] = {}
        for w, f in freq.items():
            if f >= min_freq:
                try:
                    words[w] = list(w.encode("utf-8"))
                except UnicodeEncodeError:  # pragma: no cover - defensive
                    continue

        merges: List[Tuple[int, int]] = []
        next_id = 256
        while next_id < vocab_size and words:
            counts: Counter = Counter()
            for w, ids in words.items():
                f = freq[w]
                for i in range(len(ids) - 1):
                    counts[(ids[i], ids[i + 1])] += f
            if not counts:
                break
            (a, b), best = counts.most_common(1)[0]
            if best < 2:
                break
            merges.append((a, b))
            new_id = next_id
            next_id += 1
            for w, ids in list(words.items()):
                out: List[int] = []
                i = 0
                n = len(ids)
                while i < n:
                    if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                words[w] = out
        return cls(merges, specials)

    # ----------------------------------------------------------------- encode
    def _rebuild_decode_cache(self) -> None:
        cache: Dict[int, bytes] = {}
        for i in range(256):
            cache[i] = bytes([i])
        merge_index: Dict[Tuple[int, int], int] = {}
        for idx, (a, b) in enumerate(self.merges):
            cache[256 + idx] = cache.get(a, b"?") + cache.get(b, b"?")
            merge_index[(a, b)] = 256 + idx
        for i, s in enumerate(self.specials):
            cache[self.n_merges + i] = s.encode("utf-8")
        self._decode_cache = cache
        self._merge_index = merge_index

    def _rebuild_specials(self) -> None:
        """Special-token ids are always ``n_merges + index`` — stable across
        rebuilds, and never overlapping the byte/merge range."""
        self._specials_by_len = sorted(
            ((s, self.n_merges + i) for i, s in enumerate(self.specials)),
            key=lambda p: -len(p[0]),
        )

    @property
    def n_merges(self) -> int:
        return 256 + len(self.merges)

    @property
    def vocab_size(self) -> int:
        return self.n_merges + len(self.specials)

    def id_for_special(self, text: str) -> int:
        """Id of an atomic special token, or -1 if ``text`` is not one."""
        for s, i in self._specials_by_len:
            if s == text:
                return i
        return -1

    def _encode_word(self, word: str) -> List[int]:
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode("utf-8"))
        for a, b in self.merges:
            if not ids:
                break
            out: List[int] = []
            i = 0
            n = len(ids)
            mid = self._merge_index[(a, b)]
            while i < n:
                if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                    out.append(mid)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if len(self._word_cache) < 200_000:
            self._word_cache[word] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        """Lossless: every byte of the input (whitespace runs included) survives
        decode(encode(x)) == x.  Whitespace bytes are passed through raw (merges
        are only ever learned *within* words, since training splits on
        whitespace), words go through BPE.

        Atomic special tokens (control markers) are matched first, longest-first,
        and emitted as a single id.  This is what makes a control marker a
        *reachable* next-token prediction instead of a sequence of subword pieces
        that must each win the argmax race individually.
        """
        ids: List[int] = []
        i, n = 0, len(text)
        while i < n:
            j = i
            while j < n and text[j].isspace():
                j += 1
            if j > i:
                ids.extend(text[i:j].encode("utf-8"))   # exact whitespace run
            if j >= n:
                break
            k = j
            while k < n and not text[k].isspace():
                # stop the word at a special-token boundary, so markers glued to
                # surrounding text (``a<|end|>b``) are still matched atomically
                if k > j and self._match_special(text, k) is not None:
                    break
                k += 1
            hit = self._match_special(text, j)
            if hit is not None and j + hit[0] == k:
                ids.append(hit[1])          # the whole run is one special token
                i = k
                continue
            ids.extend(self._encode_word(text[j:k]))
            i = k
        return ids

    def _match_special(self, text: str, pos: int) -> Tuple[int, int] | None:
        """Longest special token starting exactly at ``pos`` -> (length, id)."""
        for s, sid in self._specials_by_len:
            if text.startswith(s, pos):
                return len(s), sid
        return None

    # ----------------------------------------------------------------- decode
    def decode(self, ids: Iterable[int]) -> str:
        raw = b"".join(self._decode_cache.get(int(i), b"?") for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------- io
    def add_special_tokens(self, tokens: Iterable[str]) -> List[int]:
        """Append atomic tokens; returns their ids (idempotent, order-stable).

        Existing ids never move, so checkpoints trained before this call stay
        loadable (the model's embedding matrix just needs growing by ``len()``).
        """
        for t in tokens:
            if t not in self.specials:
                self.specials.append(t)
        self._rebuild_decode_cache()
        self._rebuild_specials()
        return [self.id_for_special(t) for t in tokens]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"format": "gmllm-bpe-v2",
                                 "merges": self.merges,
                                 "specials": self.specials}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in data["merges"]], data.get("specials", []))
