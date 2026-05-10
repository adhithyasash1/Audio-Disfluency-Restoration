"""Per-disfluency Whisper log-prob thresholds.

More negative = more lenient (insert even with low confidence).
Less negative = stricter (only insert if ASR is very confident).
"""

from __future__ import annotations

from .text import norm

DEFAULT_THRESHOLDS: dict[str, float] = {
    # Almost-always-fillers — be lenient.
    "हम्म": -8.0,
    "उम्म": -7.0,
    "अं": -7.0,
    "ह": -7.0,
    "अह": -7.0,
    "उह": -7.0,
    "ओ": -6.0,
    # Words that can be real — be stricter.
    "हां": -5.0,
    "हाँ": -5.0,
    "तो": -4.0,
    "वो": -4.0,
    "और": -3.0,
}

DEFAULT_FALLBACK = -6.0


def get_threshold(token: str, table: dict[str, float] | None = None) -> float:
    table = table if table is not None else DEFAULT_THRESHOLDS
    return table.get(norm(token), DEFAULT_FALLBACK)
