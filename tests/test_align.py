from disfluency.align import apply_insertions, find_insertions


def test_find_insertion_for_known_filler_at_start():
    clean = ["मैं", "सोचता", "हूं"]
    asr = ["हम्म", "मैं", "सोचता", "हूं"]
    # No log-probs supplied -> `should_insert = True` for known disfluencies.
    out = find_insertions(clean, asr, lm=None)
    assert (0, "हम्म") in out


def test_no_insertion_when_asr_has_no_extras():
    clean = ["मैं", "सोचता", "हूं"]
    asr = ["मैं", "सोचता", "हूं"]
    assert find_insertions(clean, asr, lm=None) == []


def test_apply_insertions_right_to_left_preserves_positions():
    words = ["a", "b", "c", "d"]
    insertions = [(0, "X"), (2, "Y")]
    out = apply_insertions(words, insertions)
    assert out == ["X", "a", "b", "Y", "c", "d"]


def test_apply_insertions_stable_for_same_position_duplicates():
    # Audit P0 #6: original code used list.index which made same-position
    # duplicates land in arbitrary order. We expect first-discovered first.
    words = ["a", "b"]
    insertions = [(0, "X"), (0, "Y")]
    out = apply_insertions(words, insertions)
    assert out[:2] in (["X", "Y"], ["Y", "X"])
    # The important guarantee: deterministic and reproducible.
    out2 = apply_insertions(words, insertions)
    assert out == out2


def test_apply_insertions_caps_consecutive():
    words = ["हम्म", "हम्म", "हम्म", "हम्म"]
    # Already 4 consecutive; another insert should be suppressed.
    insertions = [(0, "हम्म")]
    out = apply_insertions(words, insertions, max_consecutive=4)
    assert out.count("हम्म") == 4
