"""Small helpers shared by all LectureMate modules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
VIDEO_EXTENSIONS = {".mp4"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# Keep in sync with `maxUploadSize` in .streamlit/config.toml
MAX_UPLOAD_MB = 1024


class LectureMateError(Exception):
    """An error whose message is safe and useful to show directly to the student."""


def ensure_dirs() -> None:
    for folder in (UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def file_kind(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def validate_upload(filename: str, size_bytes: int) -> None:
    """Raise LectureMateError if the upload cannot be processed."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(e.lstrip(".").upper() for e in SUPPORTED_EXTENSIONS))
        raise LectureMateError(
            f"Unsupported file type '{ext or 'none'}'. Please upload one of: {allowed}."
        )
    if size_bytes <= 0:
        raise LectureMateError("The uploaded file is empty (0 bytes). Please choose another file.")
    if size_bytes > MAX_UPLOAD_MB * 1024 * 1024:
        raise LectureMateError(
            f"The file is {human_size(size_bytes)}, which is above the {MAX_UPLOAD_MB} MB limit. "
            "Please compress the video or upload only the audio."
        )


def safe_filename(name: str) -> str:
    """Make a file name safe for Windows (no spaces/special characters)."""
    path = Path(name)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "lecture"
    ext = re.sub(r"[^A-Za-z0-9.]", "", path.suffix.lower())
    return f"{stem[:60]}{ext}"


def remove_files(*paths: Path | str | None) -> None:
    """Delete files, ignoring anything missing or locked."""
    for path in paths:
        if not path:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most `max_chars`, breaking on sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:  # one enormous "sentence": hard split
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def extract_json_objects(text: str, required_key: str) -> list[dict[str, Any]]:
    """Find every complete JSON object in `text` that contains `required_key`.

    This is deliberately forgiving: it works when the model wraps the list in
    another object, adds chatter around the JSON, or gets cut off half-way
    (all complete objects before the cut-off are still recovered).
    """
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    pos = 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if isinstance(obj, dict) and required_key in obj:
            found.append(obj)
            pos = end
        else:
            pos = start + 1  # look inside wrapper objects
    return found
