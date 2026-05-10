"""Per-id sharded JSON cache for ASR results.

Replaces the original single-pickle cache (audit P0 #9 / P2 #9). Each audio
id gets its own JSON file under `cache_dir/`; corrupted entries fail loudly
and locally instead of taking down the whole run.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional

from .asr import TranscriptionResult, WordLogprob


class AsrCache:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, audio_id: str) -> str:
        # Avoid path traversal.
        safe = audio_id.replace(os.sep, "_").replace("..", "_")
        return os.path.join(self.cache_dir, f"{safe}.json")

    def get(self, audio_id: str) -> Optional[TranscriptionResult]:
        path = self._path(audio_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        words = [WordLogprob(**w) for w in data.get("words", [])]
        return TranscriptionResult(text=data.get("text", ""), words=words)

    def put(self, audio_id: str, result: TranscriptionResult) -> None:
        path = self._path(audio_id)
        payload = {"text": result.text, "words": [asdict(w) for w in result.words]}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, path)
