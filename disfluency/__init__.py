"""Hindi disfluency restoration package.

Pure-Python pieces (text, align, position, ngram) have no torch/transformers
dependency and are unit-tested. The ASR layer (`disfluency.asr`) is imported
lazily so that the package can be installed and tested without GPU stack.
"""

from .text import (
    DEFAULT_FILLERS,
    build_disfluency_set,
    is_disfluency,
    make_clean,
    normalize_text,
    tokenize,
)
from .thresholds import DEFAULT_THRESHOLDS, get_threshold
from .position import position_prior
from .ngram import NgramLM
from .align import apply_insertions, find_insertions

__all__ = [
    "DEFAULT_FILLERS",
    "DEFAULT_THRESHOLDS",
    "NgramLM",
    "apply_insertions",
    "build_disfluency_set",
    "find_insertions",
    "get_threshold",
    "is_disfluency",
    "make_clean",
    "normalize_text",
    "position_prior",
    "tokenize",
]

__version__ = "0.2.0"
