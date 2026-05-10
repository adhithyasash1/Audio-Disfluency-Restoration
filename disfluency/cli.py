"""Command-line entry point: `disfluency-restore` (configured in pyproject)."""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

from .cache import AsrCache
from .ngram import NgramLM
from .pipeline import RestorationConfig, Restorer


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hindi disfluency restoration pipeline.")
    p.add_argument("--audio-dir", required=True, help="Directory of {id}.wav files.")
    p.add_argument("--test-csv", required=True, help="CSV with `id, transcript`.")
    p.add_argument("--train-csv", default=None, help="CSV with `transcript` col for n-gram LM.")
    p.add_argument("--out", default="outputs/submission.csv", help="Output CSV path.")
    p.add_argument("--cache-dir", default="outputs/asr_cache", help="ASR cache dir.")
    p.add_argument("--model", default=None, help="Override Whisper model id.")
    p.add_argument("--pos-exponent", type=float, default=1.5)
    p.add_argument("--lm-threshold", type=float, default=-2.0)
    p.add_argument("--no-lm", action="store_true", help="Disable n-gram LM filtering.")
    p.add_argument("--seed", type=int, default=0)
    return p


def _set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _set_seeds(args.seed)

    if not os.path.exists(args.test_csv):
        print(f"error: test csv not found: {args.test_csv}", file=sys.stderr)
        return 2
    if not os.path.isdir(args.audio_dir):
        print(f"error: audio dir not found: {args.audio_dir}", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Build LM (optional, fast).
    lm = None
    if not args.no_lm and args.train_csv and os.path.exists(args.train_csv):
        train_df = pd.read_csv(args.train_csv)
        lm = NgramLM.from_transcripts(train_df["transcript"].dropna().astype(str).tolist())
        print(f"[lm] built trigram LM, vocab={lm.vocab_size}, ngrams={len(lm.ngram_counts)}")
    elif not args.no_lm:
        print("[lm] no --train-csv; LM filtering disabled")

    # Heavy import: keep behind the args check so `--help` is fast.
    from .asr import DEFAULT_MODEL_ID, WhisperASR

    print(f"[asr] loading {args.model or DEFAULT_MODEL_ID} ...")
    asr = WhisperASR(model_id=args.model or DEFAULT_MODEL_ID)
    cache = AsrCache(args.cache_dir)
    restorer = Restorer(
        asr=asr,
        lm=lm,
        cache=cache,
        config=RestorationConfig(
            pos_exponent=args.pos_exponent,
            use_lm=not args.no_lm and lm is not None,
            lm_threshold=args.lm_threshold,
        ),
    )

    test_df = pd.read_csv(args.test_csv)
    rows = []
    t0 = time.time()
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="restore"):
        audio_id = str(row["id"])
        audio_path = os.path.join(args.audio_dir, f"{audio_id}.wav")
        clean = "" if pd.isna(row.get("transcript")) else str(row["transcript"])
        if not os.path.exists(audio_path):
            rows.append({"id": audio_id, "transcript": clean})
            continue
        restored = restorer.restore(clean, audio_id, audio_path)
        rows.append({"id": audio_id, "transcript": restored})

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[done] wrote {args.out}  ({len(rows)} rows, {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
