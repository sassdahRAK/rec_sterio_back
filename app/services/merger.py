"""Alignment + reconciliation of two transcripts (feature 4).

Pure functions, no I/O, no database, no network - so the whole algorithm is
unit-testable with synthetic inputs.

Design
------
1. Both transcripts are flattened into *tokens*. Latin words are one token;
   Khmer runs are split per character because Khmer is written without spaces
   between words, so word-level alignment would fail on Khmer text.
2. A similarity matrix is computed with ``difflib.SequenceMatcher`` over the
   token sequences of every (base segment, candidate segment) pair. Timestamps
   are deliberately NOT used - Gemini timestamps are unreliable.
3. A global (Needleman-Wunsch style) alignment over the segment sequences turns
   the matrix into a monotonic alignment of base segments to candidate
   segments, allowing gaps on either side ("Gemini dropped this region",
   "Whisper garbled this region").
4. Each aligned region is reconciled into one output segment with a
   provenance label (``whisper`` / ``gemini`` / ``both``), a similarity score,
   a human readable reason and a ``flagged`` marker for unalignable regions.
5. The result is deterministic: the same inputs always produce the same
   output, and nothing is ever silently dropped.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- alignment tuning -------------------------------------------------------
MATCH_OFFSET = 0.75  # match score = similarity - MATCH_OFFSET
GAP_PENALTY = 0.35  # cost of leaving a segment unaligned
MIN_AGREEMENT = 0.80  # similarity above which the two transcripts "agree"
CONFLICT_MARGIN = 0.05  # quality difference below which we keep the base text
MAX_CELLS = 4_000_000  # guard against enormous matrices

# Khmer block
_KM = "\u1780-\u17ff"
_TOKEN_RE = re.compile(rf"[\u1780-\u17ff]|[^\W_]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_REPEAT_RE = re.compile(r"^(.)\1{2,}$", re.UNICODE)


def normalize_text(text: str) -> str:
    """Casefold + strip punctuation + collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(rf"[^\w\s{_KM}]+", " ", text, flags=re.UNICODE)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokens used for alignment (Khmer is split per character, see module doc)."""
    return _TOKEN_RE.findall(normalize_text(text))


def similarity(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> float:
    """Token-level similarity in [0, 1]."""
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return SequenceMatcher(None, list(a_tokens), list(b_tokens), autojunk=False).ratio()


def is_garbled(text: str) -> bool:
    """Heuristic: is this text empty / noise / a repetition loop?"""
    t = (text or "").strip()
    if not t:
        return True
    if "\ufffd" in t:
        return True
    if not any(ch.isalpha() for ch in t):
        return True
    tokens = tokenize(t)
    if not tokens:
        return True
    counts = Counter(tokens)
    top, n = counts.most_common(1)[0]
    if n >= 3 and n >= 0.6 * len(tokens):
        return True
    if len(tokens) == 1 and _REPEAT_RE.match(tokens[0]):
        return True
    return False


def quality_score(text: str) -> float:
    """Higher is better. Used to break ties between two clean, differing texts."""
    t = (text or "").strip()
    if not t:
        return 0.0
    tokens = tokenize(t)
    if not tokens:
        return 0.0
    letterish = sum(1 for tok in tokens if any(c.isalpha() for c in tok))
    letter_ratio = letterish / len(tokens)
    diversity = len(set(tokens)) / len(tokens)
    score = 0.6 * letter_ratio + 0.4 * diversity
    if len(tokens) < 2:
        score -= 0.2
    return score


@dataclass
class SegmentInput:
    """A segment as fed into the merger."""

    start: float = 0.0
    end: float = 0.0
    text: str = ""
    language: Optional[str] = None


@dataclass
class AlignedPair:
    base_index: Optional[int] = None
    gemini_index: Optional[int] = None
    similarity: float = 0.0

    @property
    def kind(self) -> str:
        if self.base_index is not None and self.gemini_index is not None:
            return "aligned"
        if self.base_index is not None:
            return "base_only"
        return "gemini_only"


@dataclass
class MergeResult:
    segments: List[Dict[str, Any]] = field(default_factory=list)
    content: str = ""
    summary: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)
    alignment: List[AlignedPair] = field(default_factory=list)


# --------------------------------------------------------------------------
# similarity matrix + global alignment
# --------------------------------------------------------------------------


def similarity_matrix(
    base_segments: Sequence[SegmentInput], gemini_segments: Sequence[SegmentInput]
) -> List[List[float]]:
    n, m = len(base_segments), len(gemini_segments)
    base_tokens = [tokenize(s.text) for s in base_segments]
    gem_tokens = [tokenize(s.text) for s in gemini_segments]
    matrix = [[0.0] * m for _ in range(n)]
    for i in range(n):
        bt = base_tokens[i]
        for j in range(m):
            matrix[i][j] = similarity(bt, gem_tokens[j])
    return matrix


def _match_score(sim: float) -> float:
    return sim - MATCH_OFFSET


def align_segments(
    base_segments: Sequence[SegmentInput], gemini_segments: Sequence[SegmentInput]
) -> List[AlignedPair]:
    """Global monotonic alignment of base segments to candidate segments."""
    n, m = len(base_segments), len(gemini_segments)
    if n == 0 and m == 0:
        return []
    if n == 0:
        return [AlignedPair(gemini_index=j) for j in range(m)]
    if m == 0:
        return [AlignedPair(base_index=i) for i in range(n)]
    if n * m > MAX_CELLS:
        # Degenerate case: align by position only, flag everything.
        pairs: List[AlignedPair] = []
        for i in range(n):
            pairs.append(AlignedPair(base_index=i))
        for j in range(m):
            pairs.append(AlignedPair(gemini_index=j))
        return pairs

    base_tokens = [tokenize(s.text) for s in base_segments]
    gem_tokens = [tokenize(s.text) for s in gemini_segments]
    matrix = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            matrix[i][j] = similarity(base_tokens[i], gem_tokens[j])

    def match_score(i: int, j: int) -> float:
        # An empty segment on one side is a neutral wildcard: the alignment
        # should still pair it up (that is the "Whisper produced nothing
        # here, Gemini did" case) instead of dropping it. A weak positional
        # prior breaks the tie in favour of the neighbouring segment.
        if not base_tokens[i] or not gem_tokens[j]:
            return -0.05 * abs(i - j)
        return matrix[i][j] - MATCH_OFFSET

    # dp[i][j] = best score aligning base[:i] with gem[:j]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - GAP_PENALTY
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - GAP_PENALTY

    for i in range(1, n + 1):
        row = dp[i]
        prev = dp[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + match_score(i - 1, j - 1)
            up = prev[j] - GAP_PENALTY
            left = row[j - 1] - GAP_PENALTY
            best = diag
            if up > best:
                best = up
            if left > best:
                best = left
            row[j] = best

    pairs: List[AlignedPair] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diag = dp[i - 1][j - 1] + match_score(i - 1, j - 1)
            if abs(dp[i][j] - diag) < 1e-9:
                pairs.append(
                    AlignedPair(
                        base_index=i - 1, gemini_index=j - 1, similarity=matrix[i - 1][j - 1]
                    )
                )
                i -= 1
                j -= 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] - GAP_PENALTY)) < 1e-9:
            pairs.append(AlignedPair(base_index=i - 1))
            i -= 1
            continue
        if j > 0:
            pairs.append(AlignedPair(gemini_index=j - 1))
            j -= 1
            continue
        break
    pairs.reverse()
    return pairs


# --------------------------------------------------------------------------
# reconciliation
# --------------------------------------------------------------------------


def _detect_language(text: str) -> Optional[str]:
    from app.services.asr import detect_language

    return detect_language(text)


def reconcile_pair(
    base: Optional[SegmentInput],
    gemini: Optional[SegmentInput],
    similarity_score: float = 0.0,
) -> Dict[str, Any]:
    """Pick the better text for one aligned region and describe why."""
    base_text = (base.text or "").strip() if base is not None else ""
    gem_text = (gemini.text or "").strip() if gemini is not None else ""

    if base is None:
        # Gemini produced a region Whisper never had.
        return {
            "text": gem_text,
            "source": "gemini",
            "language": (gemini.language if gemini else None) or _detect_language(gem_text),
            "agreement": 0.0,
            "reason": "whisper_missing",
            "flagged": True,
        }
    if gemini is None:
        # Gemini dropped the region - keep the Whisper text and flag it.
        return {
            "text": base_text,
            "source": "whisper",
            "language": (base.language if base else None) or _detect_language(base_text),
            "agreement": 0.0,
            "reason": "gemini_missing",
            "flagged": True,
        }

    if not base_text and not gem_text:
        return {
            "text": "",
            "source": "both",
            "language": base.language or gemini.language,
            "agreement": 1.0,
            "reason": "both_empty",
            "flagged": False,
        }

    base_norm = normalize_text(base_text)
    gem_norm = normalize_text(gem_text)
    if base_norm and base_norm == gem_norm:
        # They agree - keep either (we keep the base text).
        return {
            "text": base_text,
            "source": "both",
            "language": base.language or gemini.language or _detect_language(base_text),
            "agreement": 1.0,
            "reason": "identical",
            "flagged": False,
        }

    base_garbled = is_garbled(base_text)
    gem_garbled = is_garbled(gem_text)

    if base_garbled and not gem_garbled:
        return {
            "text": gem_text,
            "source": "gemini",
            "language": gemini.language or _detect_language(gem_text),
            "agreement": similarity_score,
            "reason": "whisper_garbled",
            "flagged": False,
        }
    if gem_garbled and not base_garbled:
        return {
            "text": base_text,
            "source": "whisper",
            "language": base.language or _detect_language(base_text),
            "agreement": similarity_score,
            "reason": "gemini_garbled",
            "flagged": False,
        }
    if base_garbled and gem_garbled:
        return {
            "text": base_text,
            "source": "whisper",
            "language": base.language,
            "agreement": similarity_score,
            "reason": "both_garbled",
            "flagged": True,
        }

    # The two transcripts say (almost) the same thing - keep either.
    if similarity_score >= MIN_AGREEMENT:
        return {
            "text": base_text,
            "source": "both",
            "language": base.language or gemini.language or _detect_language(base_text),
            "agreement": similarity_score,
            "reason": "agreement",
            "flagged": False,
        }

    # Both are readable but they disagree: compare quality.
    q_base = quality_score(base_text)
    q_gem = quality_score(gem_text)
    if abs(q_base - q_gem) < CONFLICT_MARGIN:
        return {
            "text": base_text,
            "source": "whisper",
            "language": base.language or _detect_language(base_text),
            "agreement": similarity_score,
            "reason": "conflict_keep_base",
            "flagged": True,
        }
    if q_base > q_gem:
        return {
            "text": base_text,
            "source": "whisper",
            "language": base.language or _detect_language(base_text),
            "agreement": similarity_score,
            "reason": "conflict_whisper_better",
            "flagged": True,
        }
    return {
        "text": gem_text,
        "source": "gemini",
        "language": gemini.language or _detect_language(gem_text),
        "agreement": similarity_score,
        "reason": "conflict_gemini_better",
        "flagged": True,
    }


def merge_transcripts(
    base_segments: Sequence[SegmentInput],
    candidate_segments: Sequence[SegmentInput],
    base_label: str = "whisper",
    candidate_label: str = "gemini",
) -> MergeResult:
    """Merge two transcripts into one, with per-segment provenance.

    The base transcript is the primary one (Whisper). The candidate is the
    second opinion (Gemini). Nothing is ever dropped: regions that only exist
    in one side are kept and flagged.
    """
    pairs = align_segments(base_segments, candidate_segments)

    out_segments: List[Dict[str, Any]] = []
    stats = {
        "base_segments": len(base_segments),
        "candidate_segments": len(candidate_segments),
        "aligned": 0,
        "identical": 0,
        "from_base": 0,
        "from_candidate": 0,
        "base_only": 0,
        "candidate_only": 0,
        "flagged": 0,
        "mean_agreement": 0.0,
    }
    agreements: List[float] = []

    for pair in pairs:
        base = base_segments[pair.base_index] if pair.base_index is not None else None
        cand = (
            candidate_segments[pair.gemini_index] if pair.gemini_index is not None else None
        )
        decision = reconcile_pair(base, cand, pair.similarity)

        if base is not None and cand is not None:
            stats["aligned"] += 1
            agreements.append(pair.similarity)
            if decision["reason"] == "identical":
                stats["identical"] += 1
        elif base is not None:
            stats["base_only"] += 1
        else:
            stats["candidate_only"] += 1

        if decision["source"] == "whisper":
            stats["from_base"] += 1
        elif decision["source"] == "gemini":
            stats["from_candidate"] += 1
        if decision["flagged"]:
            stats["flagged"] += 1

        # Timing comes from the base (Whisper) whenever it exists: Gemini
        # timestamps are not trustworthy.
        if base is not None:
            start, end = base.start, base.end
        elif cand is not None:
            start, end = cand.start, cand.end
        else:  # pragma: no cover - defensive
            start, end = 0.0, 0.0

        out_segments.append(
            {
                "index": len(out_segments),
                "start": float(start),
                "end": float(end),
                "text": decision["text"],
                "language": decision["language"],
                "source": decision["source"],
                "agreement": round(float(decision["agreement"]), 4),
                "reason": decision["reason"],
                "flagged": bool(decision["flagged"]),
                "base_index": pair.base_index,
                "candidate_index": pair.gemini_index,
                "base_text": base.text if base is not None else None,
                "candidate_text": cand.text if cand is not None else None,
            }
        )

    stats["mean_agreement"] = round(sum(agreements) / len(agreements), 4) if agreements else 0.0
    stats["output_segments"] = len(out_segments)
    stats["base_label"] = base_label
    stats["candidate_label"] = candidate_label

    content = " ".join(s["text"].strip() for s in out_segments if s["text"].strip())

    summary = _build_summary(stats)
    return MergeResult(
        segments=out_segments, content=content, summary=summary, stats=stats, alignment=pairs
    )


def _build_summary(stats: Dict[str, Any]) -> str:
    parts = [
        f"Aligned {stats['aligned']} of {stats['base_segments']} base and "
        f"{stats['candidate_segments']} candidate segments.",
        f"{stats['identical']} identical,",
        f"{stats['from_base']} taken from the base transcript,",
        f"{stats['from_candidate']} taken from the candidate transcript.",
    ]
    if stats["base_only"]:
        parts.append(
            f"{stats['base_only']} region(s) only in the base transcript were kept and flagged."
        )
    if stats["candidate_only"]:
        parts.append(
            f"{stats['candidate_only']} region(s) only in the candidate transcript were kept and flagged."
        )
    if stats["flagged"]:
        parts.append(f"{stats['flagged']} segment(s) flagged for review.")
    parts.append(f"Mean token agreement {stats['mean_agreement']}.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# convenience helpers for the worker
# --------------------------------------------------------------------------


def segments_from_rows(rows: Any) -> List[SegmentInput]:
    """Build SegmentInput list from transcript.segments jsonb rows."""
    out: List[SegmentInput] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        out.append(
            SegmentInput(
                start=float(row.get("start") or 0.0),
                end=float(row.get("end") or 0.0),
                text=str(row.get("text") or ""),
                language=row.get("language"),
            )
        )
    return out


def merge_result_to_diff(result: MergeResult) -> Dict[str, Any]:
    """The jsonb payload stored in `merges.diff`."""
    return {
        "segments": result.segments,
        "summary": result.summary,
        "stats": result.stats,
    }
