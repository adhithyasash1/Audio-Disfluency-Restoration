"""Text normalization and disfluency vocabulary utilities.

Fixes P0 #1 from the audit: Python's `\b` is ASCII-only and does not match a
boundary between whitespace and a Devanagari letter, so the original
notebook's disfluency-stripping regex silently never fired. We replace the
`\b` anchors with explicit lookarounds against whitespace and punctuation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# Hindi/Devanagari + ASCII punctuation that legitimately separates words.
_BOUNDARY_PUNCT = r"\s।॥,.!?;:'\"()\[\]\-—" + "‍‌"

DEFAULT_FILLERS: frozenset[str] = frozenset({
    "अं", "उं", "ऊं", "आं", "एं", "ओं",
    "हम्म", "हां", "हाँ",
    "उम्म", "अम्म",
    "ह", "अ", "ए",
    "तो", "वो", "जो",
    "मतलब", "बस", "अच्छा",
})


def normalize_text(text: str) -> str:
    """Lowercase, NFKC, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[।॥,.!?;:'\"()\[\]\-—]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    """Whitespace tokenization on top of `normalize_text`."""
    return [t for t in normalize_text(text).split() if t]


def norm(token: str) -> str:
    return unicodedata.normalize("NFKC", str(token).strip().lower())


def build_disfluency_set(extra: Iterable[str] | None = None) -> set[str]:
    """Combine DEFAULT_FILLERS with an optional extra iterable (e.g. CSV column)."""
    out = {norm(x) for x in DEFAULT_FILLERS}
    if extra:
        out.update(norm(x) for x in extra if x and str(x).strip())
    return out


def _build_disfluency_regex(disfluencies: Iterable[str]) -> re.Pattern[str]:
    """Build a Devanagari-safe disfluency-matching regex.

    Boundary is "start-of-string OR preceded by punctuation/whitespace" and the
    same on the right. Sorted longest-first so multi-char fillers win.
    """
    items = sorted({norm(x) for x in disfluencies if x}, key=len, reverse=True)
    if not items:
        # Match-nothing pattern.
        return re.compile(r"(?!x)x")
    body = "|".join(re.escape(x) for x in items)
    pattern = rf"(?:^|(?<=[{_BOUNDARY_PUNCT}]))(?:{body})(?=[{_BOUNDARY_PUNCT}]|$)"
    return re.compile(pattern, flags=re.UNICODE)


def make_clean(text: str, disfluencies: Iterable[str] | None = None) -> str:
    """Strip every occurrence of any disfluency token from `text`."""
    if not isinstance(text, str) or not text:
        return ""
    if disfluencies is None:
        disfluencies = DEFAULT_FILLERS
    pattern = _build_disfluency_regex(disfluencies)
    t = unicodedata.normalize("NFKC", text)
    t = pattern.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_disfluency(token: str, vocab: Iterable[str] | None = None) -> bool:
    if vocab is None:
        vocab = DEFAULT_FILLERS
    return norm(token) in {norm(x) for x in vocab}
