"""Local ASR (Whisper) + audio preparation.

The ASR engine is pluggable so it can be unit tested with a fake engine and so
that a missing `faster-whisper` install degrades to `openai-whisper` (and then
to a clear, non-crashing error).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Protocol

from app.config import settings

logger = logging.getLogger("recsterio.asr")

KHMER_START = "\u1780"
KHMER_END = "\u17ff"


class AsrUnavailable(Exception):
    """No ASR backend is installed / usable."""


@dataclass
class ASRSegment:
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    language: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "start": float(self.start),
            "end": float(self.end),
            "text": self.text,
            "language": self.language,
        }


class ASREngine(Protocol):
    name: str

    def transcribe(self, audio_path: Path) -> List[ASRSegment]:  # pragma: no cover - protocol
        ...


def detect_language(text: str) -> Optional[str]:
    """Best-effort per-segment language tag for Khmer/English code-switching."""
    if not text:
        return None
    has_km = any(KHMER_START <= ch <= KHMER_END for ch in text)
    has_latin = any("a" <= ch.lower() <= "z" for ch in text)
    if has_km and has_latin:
        return "km-en"
    if has_km:
        return "km"
    if has_latin:
        return "en"
    return None


def _load_wav_16k_mono(path: Path) -> Optional[Any]:
    """Read a 16 kHz mono 16-bit PCM wav into float32 samples in [-1, 1]."""
    try:
        import wave

        import numpy as np

        with wave.open(str(path), "rb") as wf:
            if (
                wf.getnchannels() != 1
                or wf.getsampwidth() != 2
                or wf.getframerate() != 16000
            ):
                return None
            frames = wf.readframes(wf.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:  # pragma: no cover - non-wav input falls back to the path
        logger.debug("could not read %s as wav; falling back to path", path, exc_info=True)
        return None


class FasterWhisperEngine:
    name = "faster-whisper"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        from faster_whisper import WhisperModel  # type: ignore

        self._model = WhisperModel(
            model_name or settings.whisper_model,
            device=device or settings.whisper_device,
            compute_type=compute_type or settings.whisper_compute_type,
        )

    def transcribe(self, audio_path: Path) -> List[ASRSegment]:
        # faster-whisper decodes files through PyAV, whose API drifts across
        # releases (PyAV 19 dropped the `metadata_errors` kwarg that
        # faster-whisper 1.2.x still passes). Our extract_audio step already
        # produces 16 kHz mono PCM, so decode it with the stdlib and hand the
        # model raw float32 samples to avoid depending on PyAV at all.
        audio: Any = _load_wav_16k_mono(audio_path)
        if audio is None:
            audio = str(audio_path)
        segments, _info = self._model.transcribe(
            audio,
            beam_size=5,
            vad_filter=True,
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        out: List[ASRSegment] = []
        for seg in segments:
            text = (getattr(seg, "text", "") or "").strip()
            if not text:
                continue
            out.append(
                ASRSegment(
                    start=float(getattr(seg, "start", 0.0) or 0.0),
                    end=float(getattr(seg, "end", 0.0) or 0.0),
                    text=text,
                    language=detect_language(text),
                )
            )
        return out


class OpenAIWhisperEngine:
    name = "openai-whisper"

    def __init__(self, model_name: str | None = None) -> None:
        import whisper  # type: ignore

        self._model = whisper.load_model(model_name or settings.whisper_model)

    def transcribe(self, audio_path: Path) -> List[ASRSegment]:
        result = self._model.transcribe(str(audio_path), fp16=False)
        out: List[ASRSegment] = []
        for seg in result.get("segments", []) or []:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            out.append(
                ASRSegment(
                    start=float(seg.get("start", 0.0) or 0.0),
                    end=float(seg.get("end", 0.0) or 0.0),
                    text=text,
                    language=detect_language(text),
                )
            )
        return out


def get_engine(preferred: str | None = None) -> ASREngine:
    """Return the best available local ASR engine."""
    order = [preferred] if preferred else []
    order += ["faster-whisper", "openai-whisper"]
    errors: list[str] = []
    for name in order:
        if not name:
            continue
        try:
            if name == "faster-whisper":
                return FasterWhisperEngine()
            if name == "openai-whisper":
                return OpenAIWhisperEngine()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise AsrUnavailable(
        "No local ASR engine is available. Install faster-whisper (preferred) or "
        "openai-whisper. " + " | ".join(errors)
    )


def transcribe_audio(
    audio_path: Path,
    engine: Optional[ASREngine] = None,
    engine_name: str | None = None,
) -> List[ASRSegment]:
    """Transcribe a 16 kHz mono wav into segments. `engine` is injectable for tests."""
    eng = engine if engine is not None else get_engine(engine_name)
    segments = eng.transcribe(Path(audio_path))
    cleaned: List[ASRSegment] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        if seg.language is None:
            seg.language = detect_language(text)
        cleaned.append(seg)
    return cleaned


# ------------------------------------------------------------------ ffmpeg


def probe_duration(media_path: Path, timeout_seconds: int | None = None) -> float | None:
    """Return the media duration in seconds, or None if it cannot be read.

    Used by the upload path, where there is no yt-dlp info dict to take the
    duration from. Best effort: a missing ffprobe must never fail an upload.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    timeout = timeout_seconds or 60
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    raw = (proc.stdout or b"").decode("utf-8", "replace").strip()
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def extract_audio(
    media_path: Path,
    out_path: Path,
    timeout_seconds: int | None = None,
) -> Path:
    """Extract a 16 kHz mono PCM wav from a media file (ffmpeg, with timeout)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed on the server.")

    timeout = timeout_seconds or settings.ffmpeg_timeout_seconds
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Work in a temp dir first so a failed run never leaves a truncated wav behind.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir) / "audio.wav"
        cmd = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(tmp_out),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ffmpeg timed out after {timeout}s while extracting audio."
            ) from exc
        if proc.returncode != 0 or not tmp_out.exists():
            stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
            if "does not contain any stream" in stderr or "no audio" in stderr.lower():
                raise RuntimeError(
                    "the downloaded media contains no audio stream, so it cannot be "
                    "transcribed."
                )
            raise RuntimeError(f"ffmpeg failed to extract audio: {stderr[:400]}")
        out_path.write_bytes(tmp_out.read_bytes())
    return out_path


def audio_duration(audio_path: Path) -> Optional[float]:
    """Duration of a wav file, read from the header (no ffprobe needed)."""
    try:
        import wave

        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate:
                return frames / float(rate)
    except Exception:  # pragma: no cover
        return None
    return None
