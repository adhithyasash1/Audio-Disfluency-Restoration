"""Trigram (or general n-gram) language model with add-alpha smoothing.

Encapsulated as a class so callers don't depend on module globals (audit P1).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

from .text import tokenize


class NgramLM:
    def __init__(self, n: int = 3, alpha: float = 0.1) -> None:
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self.alpha = alpha
        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.prefix_counts: Counter[tuple[str, ...]] = Counter()
        self.vocab_size: int = 1

    @classmethod
    def from_transcripts(cls, transcripts: Iterable[str], n: int = 3, alpha: float = 0.1) -> "NgramLM":
        lm = cls(n=n, alpha=alpha)
        for text in transcripts:
            if not text:
                continue
            lm.update(tokenize(str(text)))
        lm._finalize_vocab()
        return lm

    def update(self, tokens: Sequence[str]) -> None:
        seq = ["<s>"] + list(tokens) + ["</s>"]
        n = self.n
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i : i + n])
            self.ngram_counts[ngram] += 1
            self.prefix_counts[ngram[:-1]] += 1

    def _finalize_vocab(self) -> None:
        vocab = {tok for ng in self.ngram_counts for tok in ng}
        self.vocab_size = max(1, len(vocab))

    def sentence_logprob(self, tokens: Sequence[str]) -> float:
        if not self.ngram_counts:
            return 0.0
        seq = ["<s>"] + list(tokens) + ["</s>"]
        n = self.n
        v = self.vocab_size
        total = 0.0
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i : i + n])
            count = self.ngram_counts.get(ngram, 0)
            prefix_count = self.prefix_counts.get(ngram[:-1], 0)
            prob = (count + self.alpha) / (prefix_count + self.alpha * v + 1e-12)
            total += math.log(prob + 1e-12)
        return total

    def insertion_is_plausible(
        self,
        tokens: Sequence[str],
        position: int,
        candidate: str,
        delta_threshold: float = -2.0,
    ) -> bool:
        """True if inserting `candidate` at `position` doesn't drop the LM score
        below `delta_threshold`."""
        if not self.ngram_counts:
            return True
        before = self.sentence_logprob(tokens)
        with_ins = list(tokens)
        with_ins.insert(min(position, len(with_ins)), candidate)
        after = self.sentence_logprob(with_ins)
        return (after - before) > delta_threshold
