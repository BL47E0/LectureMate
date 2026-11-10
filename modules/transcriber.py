"""Speech-to-text with faster-whisper (runs fully offline on the CPU)."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .utils import LectureMateError, format_timestamp

ProgressCallback = Callable[[float], None]


@dataclass
class TranscriptResult:
    text: str          # clean paragraphs, used by the LLM and the .txt download
    timestamped: str   # one line per segment with [mm:ss], for display
    duration: float    # seconds
    language: str


def transcribe(
    audio_path: Path,
    model_size: str = "base",
    language: str | None = "en",
    progress_cb: ProgressCallback | None = None,
) -> TranscriptResult:
    """Transcribe an audio file. `language=None` means auto-detect."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise LectureMateError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model = None
    try:
        try:
            # CPU + int8 keeps memory low (~0.5-1 GB) and works on every laptop.
            model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=min(8, os.cpu_count() or 4),
            )
        except Exception as exc:  # download problems, corrupt cache, ...
            raise LectureMateError(
                f"Could not load the Whisper '{model_size}' model: {exc}. "
                "The first run downloads the model (needs internet once, ~150 MB for 'base')."
            ) from exc

        try:
            segments, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=1,                      # fastest on CPU
                vad_filter=True,                  # skip silence
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,  # avoids repetition loops
            )
            duration = float(getattr(info, "duration", 0.0) or 0.0)

            paragraphs: list[str] = []
            stamped: list[str] = []
            current: list[str] = []
            current_len = 0
            last_end = 0.0

            for seg in segments:  # `segments` is a lazy generator - this is where the work happens
                piece = seg.text.strip()
                if not piece:
                    continue
                if current and (seg.start - last_end > 2.0 or current_len > 600):
                    paragraphs.append(" ".join(current))
                    current, current_len = [], 0
                current.append(piece)
                current_len += len(piece) + 1
                last_end = seg.end
                stamped.append(f"[{format_timestamp(seg.start)}] {piece}")
                if progress_cb and duration > 0:
                    progress_cb(min(seg.end / duration, 1.0))
            if current:
                paragraphs.append(" ".join(current))
        except LectureMateError:
            raise
        except Exception as exc:
            raise LectureMateError(
                f"Transcription failed: {exc}. If you uploaded a video, make sure FFmpeg is "
                "installed; if it is an audio file, check that it is not corrupted."
            ) from exc

        text = "\n\n".join(paragraphs).strip()
        if len(text.split()) < 5:
            raise LectureMateError(
                "No speech was detected in this file. Check that the recording has audible speech."
            )
        return TranscriptResult(
            text=text,
            timestamped="\n".join(stamped),
            duration=duration,
            language=getattr(info, "language", language or "unknown"),
        )
    finally:
        model = None  # free the model's RAM before the LLM needs it
        gc.collect()
