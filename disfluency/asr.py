"""Whisper ASR wrapper.

Fixes from the audit:
- P0 #2: per-token log-probs are obtained via `model.compute_transition_scores`
  rather than the manual `out.scores` indexing that ignored the decoder prompt
  prefix in `out.sequences`.
- P0 #3: uses the modern `language=`/`task=` kwargs on `generate` instead of
  the deprecated `forced_decoder_ids` path.
- P0 #5: inference runs inside `torch.inference_mode()` instead of mutating
  the global `torch.set_grad_enabled` state.

This module imports torch/transformers/librosa lazily so the rest of the
package can be installed and tested without the GPU stack.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from typing import Iterator

from .text import normalize_text


DEFAULT_MODEL_ID = "ARTPARK-IISc/whisper-large-v3-vaani-hindi"


@dataclass
class WordLogprob:
    word: str
    avg_logprob: float | None


@dataclass
class TranscriptionResult:
    text: str
    words: list[WordLogprob] = field(default_factory=list)

    def word_logprob_map(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for w in self.words:
            if w.avg_logprob is None:
                continue
            out.setdefault(normalize_text(w.word), []).append(w.avg_logprob)
        return out


class WhisperASR:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        chunk_length_s: float = 30.0,
        max_new_tokens: int = 440,
    ) -> None:
        # Local imports keep the package importable without torch installed.
        import torch  # noqa: F401
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = __import__("torch")
        self.device = device or ("cuda" if self.torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        dtype = self.torch.float16 if self.device == "cuda" else self.torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = 16000
        self.chunk_length_s = chunk_length_s
        self.max_new_tokens = max_new_tokens

    # ---------- public API ----------

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        chunks = list(self._iter_audio_chunks(audio_path))
        if not chunks:
            return TranscriptionResult(text="", words=[])

        all_text_parts: list[str] = []
        all_words: list[WordLogprob] = []
        for chunk in chunks:
            text, words = self._transcribe_chunk(chunk)
            all_text_parts.append(text)
            all_words.extend(words)
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
        return TranscriptionResult(text=" ".join(p for p in all_text_parts if p).strip(), words=all_words)

    # ---------- internals ----------

    def _iter_audio_chunks(self, audio_path: str) -> Iterator["np.ndarray"]:  # type: ignore[name-defined]
        import librosa

        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        if len(audio) == 0:
            return
        chunk_samples = int(self.chunk_length_s * self.sample_rate)
        for start in range(0, len(audio), chunk_samples):
            yield audio[start : start + chunk_samples]

    def _transcribe_chunk(self, audio_array) -> tuple[str, list[WordLogprob]]:
        torch = self.torch
        inputs = self.processor(
            audio_array, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)
        if self.device == "cuda":
            input_features = input_features.half()

        with torch.inference_mode():
            out = self.model.generate(
                input_features,
                language="hi",
                task="transcribe",
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = out.sequences  # (B, prompt + generated)
        decoded = self.processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()

        words: list[WordLogprob]
        try:
            transition_scores = self.model.compute_transition_scores(
                sequences, out.scores, normalize_logits=True
            )  # (B, generated_len)
            generated_ids = sequences[:, sequences.shape[1] - transition_scores.shape[1] :]
            words = self._merge_subwords_to_words(
                generated_ids[0].tolist(),
                transition_scores[0].cpu().tolist(),
                decoded,
            )
        except Exception:
            # Last-resort fallback: no per-word confidences.
            words = [WordLogprob(word=w, avg_logprob=None) for w in decoded.split()]

        del input_features, out
        return decoded, words

    def _merge_subwords_to_words(
        self,
        token_ids: list[int],
        token_logprobs: list[float],
        decoded_text: str,
    ) -> list[WordLogprob]:
        tokenizer = self.processor.tokenizer
        words = decoded_text.split()
        if not words or not token_ids:
            return [WordLogprob(word=w, avg_logprob=None) for w in words]

        token_texts = [
            tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids
        ]
        out: list[WordLogprob] = []
        idx = 0
        for word in words:
            target = normalize_text(word)
            buf = ""
            lps: list[float] = []
            while idx < len(token_texts):
                buf += token_texts[idx].strip()
                if idx < len(token_logprobs):
                    lps.append(token_logprobs[idx])
                idx += 1
                buf_n = normalize_text(buf)
                if target and (target == buf_n or target in buf_n):
                    break
                if buf_n and len(buf_n) >= max(1, len(target) * 2):
                    break
            avg = float(sum(lps) / len(lps)) if lps else None
            out.append(WordLogprob(word=word, avg_logprob=avg))
        return out
