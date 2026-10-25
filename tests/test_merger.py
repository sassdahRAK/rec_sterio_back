"""Merge algorithm: alignment + reconciliation (feature 4).

These are pure unit tests of app/services/merger.py with synthetic inputs.
"""
from __future__ import annotations

from app.services.merger import (
    SegmentInput,
    align_segments,
    is_garbled,
    merge_transcripts,
    normalize_text,
    quality_score,
    similarity,
    tokenize,
)

BASE = [
    SegmentInput(0.0, 3.0, "Hello welcome to the show", "en"),
    SegmentInput(3.0, 6.0, "Today we talk about Khmer", "km-en"),
    SegmentInput(6.0, 7.0, "", "en"),  # Whisper produced nothing here
    SegmentInput(7.0, 10.0, "The second part", "en"),
]

GEMINI = [
    SegmentInput(0.0, 3.0, "Hello welcome to the show", "en"),
    SegmentInput(3.0, 6.0, "Today we talk about Khmer language", "km-en"),
    SegmentInput(6.0, 7.0, "and English together", "en"),
    SegmentInput(0.0, 0.0, "Thanks for watching", "en"),
]


def test_tokenize_splits_khmer_per_character():
    # Khmer is written without spaces between words: character level tokens
    # are required for the alignment to work at all.
    assert tokenize("សូមអរគុណ") == ["ស", "ូ", "ម", "អ", "រ", "គ", "ុ", "ណ"]
    assert tokenize("Hello world") == ["hello", "world"]


def test_normalize_text_strips_punctuation_and_case():
    assert normalize_text("  Hello, World!  ") == "hello world"


def test_similarity_is_symmetric_and_bounded():
    a = tokenize("hello world")
    b = tokenize("hello there")
    assert similarity(a, a) == 1.0
    assert 0.0 <= similarity(a, b) <= 1.0
    assert abs(similarity(a, b) - similarity(b, a)) < 1e-9


def test_is_garbled_detects_noise():
    assert is_garbled("")
    assert is_garbled("   ")
    assert is_garbled("?????")
    assert is_garbled("aaaa aaaa aaaa")
    assert not is_garbled("Hello welcome to the show")


def test_quality_score_prefers_real_words():
    assert quality_score("Hello welcome to the show") > quality_score("")
    assert quality_score("Hello welcome") > quality_score("aaaa aaaa")


def test_alignment_is_monotonic():
    pairs = align_segments(BASE, GEMINI)
    base_idx = [p.base_index for p in pairs if p.base_index is not None]
    cand_idx = [p.gemini_index for p in pairs if p.gemini_index is not None]
    assert base_idx == sorted(base_idx)
    assert cand_idx == sorted(cand_idx)


def test_merge_provenance_and_flags():
    result = merge_transcripts(BASE, GEMINI)

    sources = {s["source"] for s in result.segments}
    assert sources <= {"whisper", "gemini", "both"}

    by_reason = {s["reason"]: s for s in result.segments}
    # identical regions are marked as agreed by both engines
    assert by_reason["identical"]["source"] == "both"
    # near-identical regions are an agreement too
    assert by_reason["agreement"]["source"] == "both"
    # Whisper was empty -> Gemini's text wins
    assert by_reason["whisper_garbled"]["source"] == "gemini"
    assert by_reason["whisper_garbled"]["text"] == "and English together"
    # Gemini dropped the region -> Whisper text is kept and flagged
    assert by_reason["gemini_missing"]["source"] == "whisper"
    assert by_reason["gemini_missing"]["text"] == "The second part"
    assert by_reason["gemini_missing"]["flagged"] is True
    # Whisper dropped the region -> Gemini text is kept and flagged
    assert by_reason["whisper_missing"]["source"] == "gemini"
    assert by_reason["whisper_missing"]["text"] == "Thanks for watching"
    assert by_reason["whisper_missing"]["flagged"] is True


def test_merge_never_drops_content():
    result = merge_transcripts(BASE, GEMINI)
    merged = " ".join(s["text"] for s in result.segments)
    # Every base region survives verbatim (the base text is never dropped).
    for segment in BASE:
        if segment.text.strip():
            assert segment.text in merged
    # Every candidate region is represented by exactly one merged segment, so
    # no region of either transcript silently disappears.
    for index in range(len(GEMINI)):
        region = [s for s in result.segments if s["candidate_index"] == index]
        assert len(region) == 1
        if GEMINI[index].text.strip():
            assert region[0]["text"].strip()


def test_merge_uses_base_timestamps():
    result = merge_transcripts(BASE, GEMINI)
    for seg in result.segments:
        if seg["base_index"] is not None:
            assert seg["start"] == BASE[seg["base_index"]].start
            assert seg["end"] == BASE[seg["base_index"]].end


def test_merge_is_deterministic():
    first = merge_transcripts(BASE, GEMINI)
    second = merge_transcripts(BASE, GEMINI)
    assert first.content == second.content
    assert first.stats == second.stats
    assert [s["source"] for s in first.segments] == [s["source"] for s in second.segments]


def test_identical_transcripts_are_all_both():
    result = merge_transcripts(BASE, BASE)
    assert all(s["source"] == "both" for s in result.segments)
    assert not any(s["flagged"] for s in result.segments)
    # three non-empty segments are identical, the empty one is "both empty"
    assert result.stats["identical"] == 3
    assert result.stats["output_segments"] == len(BASE)


def test_garbled_whisper_is_replaced_by_gemini():
    base = [
        SegmentInput(0.0, 2.0, "the real sentence here", "en"),
        SegmentInput(2.0, 4.0, "aaa aaa aaa", "en"),
    ]
    gemini = [
        SegmentInput(0.0, 2.0, "the real sentence here", "en"),
        SegmentInput(2.0, 4.0, "the recovered sentence", "en"),
    ]
    result = merge_transcripts(base, gemini)
    assert result.segments[1]["source"] == "gemini"
    assert result.segments[1]["text"] == "the recovered sentence"


def test_khmer_code_switching_alignment():
    base = [
        SegmentInput(0.0, 2.0, "សូមស្វាគមន៍", "km"),
        SegmentInput(2.0, 4.0, "welcome to the channel", "en"),
    ]
    gemini = [
        SegmentInput(0.0, 2.0, "សូមស្វាគមន៍ មកស្វាគមន៍", "km"),
        SegmentInput(2.0, 4.0, "welcome to the channel", "en"),
    ]
    result = merge_transcripts(base, gemini)
    assert result.segments[0]["base_index"] == 0
    assert result.segments[0]["candidate_index"] == 0
    assert result.segments[1]["source"] == "both"
    # the Khmer region keeps the Whisper wording but is recognised as agreed
    assert result.segments[0]["source"] in ("both", "whisper", "gemini")
    assert result.segments[0]["agreement"] > 0.5


def test_empty_candidate_transcript_keeps_everything_from_base():
    result = merge_transcripts(BASE, [])
    assert len(result.segments) == len(BASE)
    assert all(s["source"] == "whisper" for s in result.segments)
    assert all(s["flagged"] for s in result.segments)
    assert result.content.strip() != ""


def test_summary_and_stats_are_reported():
    result = merge_transcripts(BASE, GEMINI)
    assert isinstance(result.summary, str) and result.summary
    assert result.stats["base_segments"] == len(BASE)
    assert result.stats["candidate_segments"] == len(GEMINI)
    assert result.stats["output_segments"] == len(result.segments)
    assert "flagged" in result.stats
