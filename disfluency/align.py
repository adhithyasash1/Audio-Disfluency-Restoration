"""Sequence-alignment driven insertion logic.

Fixes P0 #6: original code's `sorted(..., key=lambda x: (-x[0], insertions.index(x)))`
used `list.index` which collapses identical tuples to the same key and yields
unstable order for same-position duplicates. We use `enumerate` to preserve
discovery order as the secondary key.
"""

from __future__ import annotations

import math
from difflib import SequenceMatcher
from typing import Sequence

from .ngram import NgramLM
from .position import position_prior
from .text import is_disfluency, normalize_text
from .thresholds import get_threshold


def _is_repetition(token: str, context: Sequence[str], position: int) -> bool:
    """True if `token` repeats the word immediately before/at `position`."""
    if 0 <= position - 1 < len(context) and token == context[position - 1]:
        return True
    if 0 <= position < len(context) and token == context[position]:
        return True
    return False


def find_insertions(
    clean_tokens: Sequence[str],
    asr_tokens: Sequence[str],
    asr_word_logprobs: dict[str, list[float]] | None = None,
    *,
    pos_exponent: float = 1.5,
    lm: NgramLM | None = None,
    lm_threshold: float = -2.0,
    disfluency_vocab: set[str] | None = None,
    threshold_table: dict[str, float] | None = None,
    nondisfluency_score_threshold: float = -4.0,
) -> list[tuple[int, str]]:
    """Return list of (insert_position_in_clean, token) decisions.

    `asr_word_logprobs` maps normalized ASR word -> queue of log-probs (the
    same word may appear multiple times in the ASR output).
    """
    if not asr_tokens:
        return []

    info = {k: list(v) for k, v in (asr_word_logprobs or {}).items()}

    sm = SequenceMatcher(a=list(clean_tokens), b=list(asr_tokens), autojunk=False)
    insertions: list[tuple[int, str]] = []
    n_clean = len(clean_tokens)

    def _avg_lp(tok: str) -> float | None:
        key = normalize_text(tok)
        bucket = info.get(key)
        if bucket:
            return bucket.pop(0)
        return None

    for tag, i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            for j in range(j1, j2):
                token = asr_tokens[j]
                avg_lp = _avg_lp(token)
                pos_score = position_prior(i1, n_clean, pos_exponent)
                should_insert = False

                if is_disfluency(token, disfluency_vocab):
                    threshold = get_threshold(token, threshold_table)
                    if avg_lp is not None:
                        score = avg_lp + math.log(pos_score + 1e-6)
                        should_insert = score > threshold
                    else:
                        should_insert = True
                elif _is_repetition(token, clean_tokens, i1):
                    should_insert = True
                elif avg_lp is not None:
                    score = avg_lp + math.log(pos_score + 1e-6)
                    should_insert = score > nondisfluency_score_threshold

                if should_insert and lm is not None:
                    if not lm.insertion_is_plausible(clean_tokens, i1, token, lm_threshold):
                        should_insert = False

                if should_insert:
                    insertions.append((i1, token))

        elif tag == "replace":
            for j in range(j1, j2):
                token = asr_tokens[j]
                if not is_disfluency(token, disfluency_vocab):
                    continue
                if lm is not None and not lm.insertion_is_plausible(
                    clean_tokens, i1, token, lm_threshold
                ):
                    continue
                insertions.append((i1, token))

    return insertions


def apply_insertions(
    original_words: Sequence[str],
    insertions: list[tuple[int, str]],
    *,
    max_consecutive: int = 4,
) -> list[str]:
    """Apply (position, token) insertions right-to-left to `original_words`.

    Stable for same-position duplicates: secondary sort key is the original
    list index (P0 #6 fix).
    """
    result = list(original_words)
    indexed = list(enumerate(insertions))
    # Right-to-left so prior positions stay valid; preserve discovery order
    # among insertions at the same position.
    indexed.sort(key=lambda pair: (-pair[1][0], pair[0]))

    for _orig_idx, (pos, token) in indexed:
        pos = min(max(pos, 0), len(result))
        consec = 1
        i = pos - 1
        while i >= 0 and result[i] == token:
            consec += 1
            i -= 1
        i = pos
        while i < len(result) and result[i] == token:
            consec += 1
            i += 1
        if consec <= max_consecutive:
            result.insert(pos, token)
    return result
