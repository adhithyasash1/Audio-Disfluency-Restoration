from disfluency.ngram import NgramLM


def test_lm_builds_from_transcripts():
    lm = NgramLM.from_transcripts(["मैं सोचता हूं", "मैं ठीक हूं"])
    assert len(lm.ngram_counts) > 0
    assert lm.vocab_size >= 4


def test_empty_lm_returns_zero_logprob():
    lm = NgramLM()
    assert lm.sentence_logprob(["a", "b"]) == 0.0
    # Should default to "plausible" when LM is empty.
    assert lm.insertion_is_plausible(["a", "b"], 1, "x") is True


def test_insertion_plausibility_prefers_seen_ngrams():
    lm = NgramLM.from_transcripts(["हम्म मैं सोचता हूं"] * 20)
    plausible = lm.insertion_is_plausible(["मैं", "सोचता", "हूं"], 0, "हम्म")
    implausible = lm.insertion_is_plausible(["मैं", "सोचता", "हूं"], 0, "क्ष्क्ष्")
    assert plausible or not implausible  # weak ordering check; both can be True with smoothing
    # Stronger: a frequently-seen insertion gets a less-negative delta than gibberish.
    delta_seen = lm.sentence_logprob(["हम्म", "मैं", "सोचता", "हूं"]) - lm.sentence_logprob(
        ["मैं", "सोचता", "हूं"]
    )
    delta_gibberish = lm.sentence_logprob(["क्ष्क्ष्", "मैं", "सोचता", "हूं"]) - lm.sentence_logprob(
        ["मैं", "सोचता", "हूं"]
    )
    assert delta_seen > delta_gibberish
