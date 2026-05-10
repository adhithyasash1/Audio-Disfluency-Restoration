"""Regression tests for text utilities — especially the Devanagari `\\b` fix (P0 #1)."""

from disfluency.text import make_clean, normalize_text, tokenize, is_disfluency


def test_normalize_strips_punctuation_and_lowercases():
    assert normalize_text("हम्म, मैं सोचता हूं।") == "हम्म मैं सोचता हूं"


def test_tokenize_basic():
    assert tokenize("मैं   सोचता हूं") == ["मैं", "सोचता", "हूं"]


def test_make_clean_actually_strips_devanagari_filler():
    # Audit P0 #1: original notebook used \b...\b which never matched here.
    # The new boundary must remove "हम्म" but keep the rest of the sentence.
    text = "हम्म मैं सोचता हूं"
    cleaned = make_clean(text)
    assert "हम्म" not in cleaned
    assert "सोचता" in cleaned
    assert cleaned == "मैं सोचता हूं"


def test_make_clean_does_not_strip_substring_inside_word():
    # "हां" appears inside no other content word here; the regex must match
    # whole-word fillers but should not chew into adjoining letters.
    text = "हां वहां जा"  # "वहां" contains "हां" as substring
    cleaned = make_clean(text)
    assert "वहां" in cleaned, f"substring match leaked: {cleaned!r}"
    assert cleaned.startswith("वहां") or "हां वहां" not in cleaned


def test_make_clean_handles_punctuation_boundary():
    text = "हम्म, मैं ठीक हूं।"
    assert "हम्म" not in make_clean(text)


def test_is_disfluency_default_vocab():
    assert is_disfluency("हम्म")
    assert is_disfluency("तो")
    assert not is_disfluency("किताब")  # "book"


def test_make_clean_empty_safe():
    assert make_clean("") == ""
    assert make_clean(None) == ""  # type: ignore[arg-type]
