"""Gemini client: independent transcription of the audio ("second opinion").

The HTTP client is injectable so the whole module is unit-testable without an
API key. `GEMINI_API_KEY` is only ever read from the environment and is never
logged or returned.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol

from app.config import settings
from app.services.asr import ASRSegment, detect_language

logger = logging.getLogger("recsterio.gemini")

PROMPT = (
    "You are a careful transcription engine for Khmer and English audio. "
    "Transcribe the attached audio verbatim. Do not translate, do not summarise "
    "and do not add commentary. If the audio is not speech, return an empty "
    "result. Reply with JSON only, in exactly this shape:\n"
    '{"segments": [{"start": 0.0, "end": 2.5, "text": "..."}]}\n'
    "Use seconds for start/end. Timestamps are best-effort and may be omitted "
    "or set to 0.0."
)


class GeminiError(Exception):
    """Gemini call failed (missing key, API error, unusable response)."""


class GeminiClientProtocol(Protocol):
    def transcribe_audio(
        self, audio_path: Path, model: str | None = None
    ) -> List[ASRSegment]:  # pragma: no cover - protocol
        ...


@dataclass
class GeminiConfig:
    api_key: Optional[str] = None
    model: str = "gemini-2.5-pro"
    chunk_seconds: int = 600


class GeminiClient:
    """Thin wrapper over `google-genai` (or an injected stub client)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        chunk_seconds: Optional[int] = None,
        client: Any = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.chunk_seconds = chunk_seconds or settings.gemini_audio_chunk_seconds
        self._client = client

    # ------------------------------------------------------------ internals
    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise GeminiError(
                "GEMINI_API_KEY is not configured; cannot run the Gemini path."
            )
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise GeminiError(
                "google-genai is not installed on the server (pip install -r requirements.txt)."
            ) from exc
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _generate(self, audio_path: Path, model: str) -> str:
        client = self._ensure_client()
        try:
            uploaded = client.files.upload(file=str(audio_path))
        except Exception as exc:
            raise GeminiError(f"Gemini file upload failed: {exc}") from exc

        contents = [uploaded, PROMPT]
        try:
            response = client.models.generate_content(model=model, contents=contents)
        except Exception as exc:
            raise GeminiError(f"Gemini request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            candidates = getattr(response, "candidates", None) or []
            parts: list[str] = []
            for cand in candidates:
                content = getattr(cand, "content", None)
                for part in getattr(content, "parts", None) or []:
                    t = getattr(part, "text", None)
                    if t:
                        parts.append(t)
            text = "\n".join(parts)
        if not text:
            raise GeminiError("Gemini returned an empty response.")
        return text

    # ------------------------------------------------------------ public API
    def transcribe_audio(
        self, audio_path: Path, model: str | None = None
    ) -> List[ASRSegment]:
        """Transcribe an audio file with Gemini. Returns [] for silent audio."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise GeminiError(f"Audio file not found: {audio_path}")
        model = model or self.model
        raw = self._generate(audio_path, model)
        return parse_gemini_response(raw)


def parse_gemini_response(raw: str) -> List[ASRSegment]:
    """Parse a Gemini reply into segments. Tolerates fences and plain text."""
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```$", "", text).strip()

    data: Any = None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except (ValueError, TypeError):
                data = None

    segments: List[ASRSegment] = []
    if isinstance(data, dict):
        for item in data.get("segments") or []:
            if not isinstance(item, dict):
                continue
            body = str(item.get("text") or "").strip()
            if not body:
                continue
            segments.append(
                ASRSegment(
                    start=_as_float(item.get("start")),
                    end=_as_float(item.get("end")),
                    text=body,
                    language=detect_language(body),
                )
            )
        return segments

    # Fallback: treat the whole reply as one segment of plain text.
    plain = text.strip()
    if plain:
        segments.append(
            ASRSegment(start=0.0, end=0.0, text=plain, language=detect_language(plain))
        )
    return segments


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
