"""High-level orchestration: clean text + audio -> restored text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .align import apply_insertions, find_insertions
from .asr import TranscriptionResult, WhisperASR
from .cache import AsrCache
from .ngram import NgramLM
from .text import tokenize


@dataclass
class RestorationConfig:
    pos_exponent: float = 1.5
    use_lm: bool = True
    lm_threshold: float = -2.0
    max_consecutive: int = 4


class Restorer:
    def __init__(
        self,
        asr: WhisperASR,
        lm: Optional[NgramLM] = None,
        cache: Optional[AsrCache] = None,
        config: Optional[RestorationConfig] = None,
    ) -> None:
        self.asr = asr
        self.lm = lm
        self.cache = cache
        self.config = config or RestorationConfig()

    def transcribe(self, audio_id: str, audio_path: str) -> TranscriptionResult:
        if self.cache is not None:
            hit = self.cache.get(audio_id)
            if hit is not None:
                return hit
        result = self.asr.transcribe(audio_path)
        if self.cache is not None:
            self.cache.put(audio_id, result)
        return result

    def restore(self, clean_text: str, audio_id: str, audio_path: str) -> str:
        clean_text = clean_text or ""
        asr_result = self.transcribe(audio_id, audio_path)
        if not asr_result.text:
            return clean_text
        if not clean_text:
            return asr_result.text

        clean_tokens = tokenize(clean_text)
        asr_tokens = tokenize(asr_result.text)
        insertions = find_insertions(
            clean_tokens,
            asr_tokens,
            asr_word_logprobs=asr_result.word_logprob_map(),
            pos_exponent=self.config.pos_exponent,
            lm=self.lm if self.config.use_lm else None,
            lm_threshold=self.config.lm_threshold,
        )
        if not insertions:
            return clean_text

        original_words = clean_text.split()
        restored = apply_insertions(
            original_words, insertions, max_consecutive=self.config.max_consecutive
        )
        return " ".join(restored)
