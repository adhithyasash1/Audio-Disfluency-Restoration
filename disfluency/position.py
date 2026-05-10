"""Position prior: disfluencies are more likely near the start of an utterance."""

from __future__ import annotations


def position_prior(token_index: int, n_tokens: int, exponent: float = 1.5) -> float:
    """Return a value in [0, 1]; 1.0 at the start, decaying toward 0 at the end."""
    if n_tokens <= 1:
        return 1.0
    frac = token_index / float(max(1, n_tokens - 1))
    frac = min(max(frac, 0.0), 1.0)
    return 1.0 - frac ** exponent
