# 🎙️ Hindi Disfluency Restoration Pipeline

> Automatically restore filler words (disfluencies) to clean Hindi transcripts using speech recognition and language modeling.

---

## 📋 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Pipeline Overview](#pipeline-overview)
3. [Input & Output](#input--output)
4. [Models Used](#models-used)
5. [Core Logic](#core-logic)
6. [Evaluation Metric](#evaluation-metric)
7. [File Structure](#file-structure)
8. [How to Run](#how-to-run)

---

## Problem Statement

When people speak naturally, they use **filler words** like:
- English: "um", "uh", "like", "you know"
- Hindi: "हम्म" (hmm), "हां" (yeah), "उम्म" (umm), "तो" (so), "मतलब" (I mean)

These are called **disfluencies**. In many transcription datasets, these fillers are removed to create "clean" text. But for natural-sounding speech synthesis or analysis, we need them back!

**Task**: Given a clean Hindi transcript and its corresponding audio, restore the disfluencies that were originally spoken.

---

## Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Clean Text    │     │   Audio File     │     │  Disfluency     │
│   (no fillers)  │     │   (.wav)         │     │  List           │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                        │
         │                       ▼                        │
         │              ┌────────────────┐                │
         │              │  Whisper ASR   │                │
         │              │  (Hindi model) │                │
         │              └────────┬───────┘                │
         │                       │                        │
         │                       ▼                        │
         │              ┌────────────────┐                │
         │              │ ASR Output +   │                │
         │              │ Confidence     │                │
         │              └────────┬───────┘                │
         │                       │                        │
         ▼                       ▼                        ▼
    ┌────────────────────────────────────────────────────────┐
    │              Sequence Alignment (SequenceMatcher)       │
    │   Compare clean text with ASR output to find INSERT ops │
    └────────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
    ┌────────────────────────────────────────────────────────┐
    │                    Filtering Logic                      │
    │  1. Is it a known disfluency?                          │
    │  2. Does ASR confidence exceed threshold?               │
    │  3. Does position prior favor this location?            │
    │  4. Does language model approve?                        │
    └────────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
    ┌────────────────────────────────────────────────────────┐
    │              Apply Insertions to Clean Text             │
    └────────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
                      ┌──────────────────┐
                      │  Restored Text   │
                      │  (with fillers)  │
                      └──────────────────┘
```

---

## Input & Output

### Input

| File | Description |
|------|-------------|
| `test.csv` | Contains `id` and `transcript` (clean text without disfluencies) |
| `downloaded_audios/` | Audio files named `{id}.wav` |
| `unique_disfluencies.csv` | List of known Hindi disfluency words |

**Example Input Row:**
```
id: audio_001
transcript: "मैं सोचता हूं कि यह अच्छा है"
```

### Output

| File | Description |
|------|-------------|
| `submission.csv` | Contains `id` and `transcript` (restored text with disfluencies) |

**Example Output Row:**
```
id: audio_001
transcript: "हम्म मैं सोचता हूं कि यह अच्छा है"  ← "हम्म" restored!
```

---

## Models Used

### 1. Whisper Large V3 (Hindi Fine-tuned)

**What:** Automatic Speech Recognition (ASR) model by OpenAI, fine-tuned on Hindi data by ARTPARK-IISc.

**Why:** 
- State-of-the-art accuracy on Hindi speech
- Captures spoken disfluencies in its output
- Provides per-token confidence scores (log probabilities)

**How it's used:**
```python
# Load model
model = WhisperForConditionalGeneration.from_pretrained(
    "ARTPARK-IISc/whisper-large-v3-vaani-hindi"
)

# Transcribe audio with confidence scores
output = model.generate(
    audio_features,
    output_scores=True,  # Get confidence for each token
    return_dict_in_generate=True
)
```

**Model ID:** `ARTPARK-IISc/whisper-large-v3-vaani-hindi`

---

### 2. N-Gram Language Model (Trigram)

**What:** A simple statistical model that predicts word probability based on previous 2 words.

**Why:**
- Fast and lightweight
- Filters out insertions that create unnatural sentences
- Built from training transcripts (domain-specific)

**How it's used:**
```python
# Check if inserting "हम्म" sounds natural
sentence_before = "मैं सोचता हूं"           # P(sentence) = X
sentence_after = "हम्म मैं सोचता हूं"       # P(sentence) = Y

if Y - X > threshold:  # If probability doesn't drop too much
    insert_disfluency()
```

---

## Core Logic

### Step 1: Sequence Alignment

We use Python's `SequenceMatcher` to find differences between clean text and ASR output:

```python
clean_tokens = ["मैं", "सोचता", "हूं"]
asr_tokens   = ["हम्म", "मैं", "सोचता", "हूं"]

# SequenceMatcher finds:
# - INSERT "हम्म" at position 0
```

### Step 2: Per-Disfluency Thresholds

Each disfluency has its own confidence threshold:

| Disfluency | Threshold | Reason |
|------------|-----------|--------|
| हम्म (hmm) | -8.0 | Almost always a filler, be lenient |
| उम्म (umm) | -7.0 | Almost always a filler |
| हां (yes) | -5.0 | Could be real word, be stricter |
| तो (so) | -4.0 | Often real conjunction |
| और (and) | -3.0 | Almost always real word, very strict |

**Logic:**
```python
asr_confidence = -3.5  # From Whisper
threshold = -5.0       # For "हां"

if asr_confidence > threshold:  # -3.5 > -5.0 ✓
    insert_disfluency()
```

### Step 3: Position Prior

Disfluencies usually occur at the **start** of sentences (when speaker is thinking).

```python
def position_prior(position, total_words):
    # Returns score 0-1
    # Position 0 (start) → 1.0 (high)
    # Position N (end) → 0.0 (low)
    return 1.0 - (position / total_words) ** 1.5
```

The final score combines ASR confidence and position:
```python
final_score = asr_logprob + log(position_prior)
```

### Step 4: Language Model Check

Before inserting, verify the sentence still sounds natural:

```python
if check_insertion_plausibility(clean_tokens, position, "हम्म"):
    insert_disfluency()
```

---

## Evaluation Metric

### Word Error Rate (WER)

WER measures how different the predicted transcript is from the reference:

```
WER = (Substitutions + Deletions + Insertions) / Total Reference Words
```

**Example:**
```
Reference:  "हम्म मैं सोचता हूं"  (4 words)
Prediction: "मैं सोचता हूं"      (3 words)

Deletions = 1 ("हम्म" missing)
WER = 1/4 = 0.25 = 25%
```

**Lower WER = Better!**

The competition ranks submissions by WER on the test set.

---

## File Structure

```
Auto Disfluency Restoration/
├── submission.ipynb        # Main pipeline notebook
├── README.md               # This file
├── baseline.py             # Standalone Python version
├── disfluency_model_based.py  # Experimental version with calibration
└── downloaded_audios/      # Audio files (on Kaggle)
    ├── audio_001.wav
    ├── audio_002.wav
    └── ...
```

---

## How to Run

### On Kaggle

1. Create a new notebook
2. Add the competition dataset as input
3. Copy cells from `submission.ipynb`
4. Run all cells
5. Submit `submission.csv`

### Locally (for testing)

```bash
# Install dependencies
pip install transformers torch librosa jiwer pandas numpy

# Update paths in notebook to local directories
AUDIO_DIR = "./downloaded_audios"
TEST_CSV = "./test.csv"
# etc.

# Run notebook
jupyter notebook submission.ipynb
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pos_exponent` | 1.5 | Higher = stronger bias toward sentence start |
| `use_lm` | True | Whether to use language model filtering |
| `lm_threshold` | -2.0 | More negative = more lenient insertions |
| `max_consecutive` | 4 | Limit repeated same tokens |

---

## Summary

1. **Input**: Clean transcript + Audio
2. **ASR**: Whisper transcribes audio (captures disfluencies)
3. **Align**: Compare clean vs ASR to find insertions
4. **Filter**: Keep only confident, natural-sounding insertions
5. **Output**: Restored transcript with disfluencies

The pipeline balances **precision** (don't insert wrong words) with **recall** (don't miss real disfluencies) using confidence thresholds, position priors, and language model validation.
