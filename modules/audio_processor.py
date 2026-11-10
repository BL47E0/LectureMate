"""Audio preparation: extract the audio track from video files using FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .utils import LectureMateError, file_kind


def find_ffmpeg() -> str | None:
    """Return the path to ffmpeg if it is installed and on PATH."""
    return shutil.which("ffmpeg")


def extract_audio(video_path: Path, out_dir: Path) -> Path:
    """Extract a 16 kHz mono WAV (the format Whisper likes best) from a video."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise LectureMateError(
            "FFmpeg was not found. Install it (see README, 'FFmpeg installation') "
            "and restart the terminal."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_audio.wav"
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vn",                     # drop the video stream
        "-ac", "1",                # mono
        "-ar", "16000",            # 16 kHz
        "-c:a", "pcm_s16le",       # plain WAV
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=3600)
    except subprocess.TimeoutExpired as exc:
        raise LectureMateError("Audio extraction took too long and was stopped.") from exc
    except OSError as exc:
        raise LectureMateError(f"Could not run FFmpeg: {exc}") from exc

    if proc.returncode != 0:
        last_line = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        raise LectureMateError(
            f"FFmpeg could not read this video ({last_line[0]}). The file may be corrupted."
        )
    if not out_path.exists() or out_path.stat().st_size < 1024:
        raise LectureMateError("No audio track was found in this video.")
    return out_path


def prepare_audio(saved_path: Path, temp_dir: Path) -> tuple[Path, str | None]:
    """Return (path_to_give_to_whisper, optional_note_for_the_user).

    * audio files are used as they are
    * video files go through FFmpeg; if FFmpeg is missing we fall back to letting
      faster-whisper decode the video itself (it bundles its own decoder)
    """
    if file_kind(saved_path.name) != "video":
        return saved_path, None

    if find_ffmpeg() is None:
        return saved_path, (
            "FFmpeg not found - trying to read the audio straight from the video instead. "
            "Install FFmpeg for best results."
        )
    return extract_audio(saved_path, temp_dir), None
