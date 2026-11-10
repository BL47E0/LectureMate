"""Turns a transcript into study material using the local LLM.

Long lectures are handled with a simple map-reduce strategy:

    transcript -> split into chunks -> condense each chunk -> join the notes

The condensed text is then used as the context for every feature, so the model
never receives more text than its context window can hold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from . import prompts
from .llm import OllamaClient
from .utils import extract_json_objects, split_text

# Character budgets (roughly 4 characters per token; the model window is 8192 tokens)
MAX_DIRECT_CHARS = 14_000      # longest lecture text sent to the model as-is
CHUNK_CHARS = 8_000            # size of each chunk when condensing
RETRIEVAL_CHUNK_CHARS = 2_500  # size of excerpts used to answer questions on long lectures
MAX_CONDENSE_ROUNDS = 3

LETTERS = "ABCD"
ProgressCallback = Callable[[float, str], None]


@dataclass
class LectureContext:
    text: str                 # what gets sent to the LLM (raw if short, condensed if long)
    condensed: bool           # True when the lecture was too long and had to be condensed
    word_count: int
    chunks: list[str] = field(default_factory=list)  # raw excerpts, used for Q&A retrieval


def combine_sources(transcript: str, slides_text: str = "") -> str:
    parts = [f"LECTURE TRANSCRIPT:\n{transcript.strip()}"]
    if slides_text.strip():
        parts.append(f"LECTURE SLIDES (text extracted from the PDF):\n{slides_text.strip()}")
    return "\n\n".join(parts)


# ----------------------------------------------------------------- JSON helpers
def _clean_option(text: Any) -> str:
    return re.sub(r"^\s*[A-Da-d][\).:\-]\s+", "", str(text)).strip()


def _normalize_mcq(item: dict[str, Any]) -> dict[str, Any] | None:
    question = str(item.get("question", "")).strip()
    raw_options = item.get("options") or item.get("choices")
    options: list[str] = []
    if isinstance(raw_options, dict):
        lowered = {str(k).strip().upper(): v for k, v in raw_options.items()}
        options = [_clean_option(lowered.get(letter, "")) for letter in LETTERS]
    elif isinstance(raw_options, list):
        options = [_clean_option(o) for o in raw_options[:4]]
    if not question or len(options) != 4 or not all(options):
        return None

    correct = str(item.get("correct") or item.get("answer") or item.get("correct_answer") or "").strip()
    letter = None
    if correct[:1].upper() in LETTERS and not correct[1:2].isalnum():
        letter = correct[:1].upper()
    else:
        for i, option in enumerate(options):
            if option.lower() == _clean_option(correct).lower():
                letter = LETTERS[i]
                break
    if letter is None:
        return None
    return {
        "question": question,
        "options": options,
        "correct": letter,
        "explanation": str(item.get("explanation", "")).strip(),
    }


def _normalize_flashcard(item: dict[str, Any]) -> dict[str, str] | None:
    question = str(item.get("question", "")).strip()
    answer = str(item.get("answer", "")).strip()
    return {"question": question, "answer": answer} if question and answer else None


class StudyGenerator:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    # ------------------------------------------------------------- context
    def build_context(
        self,
        transcript: str,
        slides_text: str = "",
        progress_cb: ProgressCallback | None = None,
    ) -> LectureContext:
        source = combine_sources(transcript, slides_text)
        words = len(source.split())
        retrieval_chunks = split_text(source, RETRIEVAL_CHUNK_CHARS)

        if len(source) <= MAX_DIRECT_CHARS:
            return LectureContext(source, False, words, retrieval_chunks)

        text = source
        for _ in range(MAX_CONDENSE_ROUNDS):
            parts = split_text(text, CHUNK_CHARS)
            notes: list[str] = []
            for i, part in enumerate(parts, start=1):
                if progress_cb:
                    progress_cb((i - 1) / len(parts), f"Condensing long lecture: part {i} of {len(parts)}")
                system, user = prompts.condense(part, i, len(parts))
                notes.append(self.llm.chat(system, user, temperature=0.2, num_predict=900))
            text = "\n\n".join(notes)
            if len(text) <= MAX_DIRECT_CHARS:
                break
        text = text[:MAX_DIRECT_CHARS]  # safety net
        if progress_cb:
            progress_cb(1.0, "Lecture prepared")
        return LectureContext(text, True, words, retrieval_chunks)

    # ----------------------------------------------------- text generators
    def summary(self, ctx: LectureContext) -> str:
        system, user = prompts.summary(ctx.text)
        return self.llm.chat(system, user, temperature=0.3, num_predict=700)

    def notes(self, ctx: LectureContext) -> str:
        system, user = prompts.notes(ctx.text)
        return self.llm.chat(system, user, temperature=0.3, num_predict=2000)

    def concepts(self, ctx: LectureContext) -> str:
        max_n = max(6, min(15, ctx.word_count // 300 + 5))
        min_n = max(5, max_n - 5)
        system, user = prompts.concepts(ctx.text, min_n, max_n)
        return self.llm.chat(system, user, temperature=0.3, num_predict=1500)

    # ------------------------------------------------------ JSON generators
    def _json_items(
        self, system: str, user: str, required_key: str, normalizer: Callable[[dict], Any], num_predict: int
    ) -> dict[str, Any]:
        """Ask for JSON, parse it forgivingly, retry once, and never crash."""
        raw = ""
        for temperature in (0.3, 0.1):
            raw = self.llm.chat(system, user, json_mode=True, temperature=temperature, num_predict=num_predict)
            items, seen = [], set()
            for obj in extract_json_objects(raw, required_key):
                item = normalizer(obj)
                if item and item["question"].lower() not in seen:
                    seen.add(item["question"].lower())
                    items.append(item)
            if items:
                return {"items": items, "raw": None}
        return {"items": [], "raw": raw}  # fallback: the UI shows the raw text

    def mcqs(self, ctx: LectureContext, n: int = 10) -> dict[str, Any]:
        system, user = prompts.mcqs(ctx.text, n)
        result = self._json_items(system, user, "question", _normalize_mcq, 3500)
        result["items"] = result["items"][:n]
        return result

    def flashcards(self, ctx: LectureContext, n: int = 10) -> dict[str, Any]:
        system, user = prompts.flashcards(ctx.text, n)
        result = self._json_items(system, user, "question", _normalize_flashcard, 2500)
        result["items"] = result["items"][:n]
        return result

    # ----------------------------------------------------------------- Q&A
    def _qa_context(self, ctx: LectureContext, question: str) -> str:
        if not ctx.condensed:
            return ctx.text
        # Long lecture: condensed overview + the raw excerpts that best match the question.
        query = {w for w in re.findall(r"[a-z0-9]{3,}", question.lower()) if w not in _STOPWORDS}
        scored = []
        for index, chunk in enumerate(ctx.chunks):
            score = sum(1 for w in re.findall(r"[a-z0-9]{3,}", chunk.lower()) if w in query)
            if score > 0:
                scored.append((score, index))
        best = sorted(index for _, index in sorted(scored, reverse=True)[:3])
        excerpts = "\n\n".join(f"[Excerpt] {ctx.chunks[i]}" for i in best)
        text = f"OVERVIEW NOTES:\n{ctx.text[:6000]}"
        if excerpts:
            text += f"\n\nRELEVANT EXCERPTS FROM THE ORIGINAL LECTURE:\n{excerpts}"
        return text

    def answer_stream(
        self, ctx: LectureContext, question: str, history: list[dict[str, str]] | None = None
    ) -> Iterator[str]:
        system = prompts.qa_system(self._qa_context(ctx, question))
        return self.llm.stream(system, question, history=history, temperature=0.2, num_predict=1200)

    def answer(self, ctx: LectureContext, question: str, history: list[dict[str, str]] | None = None) -> str:
        return "".join(self.answer_stream(ctx, question, history))


_STOPWORDS = {
    "the", "and", "for", "are", "was", "what", "which", "who", "how", "why", "when", "where",
    "does", "did", "this", "that", "with", "from", "about", "into", "give", "tell", "explain",
    "lecture", "lecturer", "please", "can", "you", "me", "his", "her", "their", "there", "have",
    "has", "any", "some", "most", "important", "simple", "words",
}


# ------------------------------------------------------------- Markdown export
def mcqs_to_markdown(result: dict[str, Any]) -> str:
    if not result.get("items"):
        return result.get("raw") or ""
    blocks = []
    for i, q in enumerate(result["items"], start=1):
        lines = [f"Question {i}:", q["question"], ""]
        lines += [f"{LETTERS[j]}. {opt}" for j, opt in enumerate(q["options"])]
        lines += ["", f"Correct Answer: {q['correct']}", "", f"Explanation: {q['explanation']}"]
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


def flashcards_to_markdown(result: dict[str, Any]) -> str:
    if not result.get("items"):
        return result.get("raw") or ""
    return "\n\n".join(
        f"Question:\n{c['question']}\n\nAnswer:\n{c['answer']}" for c in result["items"]
    )


def to_markdown(title: str, results: dict[str, Any]) -> str:
    """Bundle everything generated so far into one Markdown document."""
    sections = [f"# LectureMate study material: {title}"]
    order = [
        ("summary", "Summary", lambda r: r),
        ("notes", "Study Notes", lambda r: r),
        ("concepts", "Key Concepts", lambda r: r),
        ("mcqs", "Quiz (MCQs)", mcqs_to_markdown),
        ("flashcards", "Flashcards", flashcards_to_markdown),
    ]
    for key, heading, render in order:
        if key in results and results[key]:
            sections.append(f"## {heading}\n\n{render(results[key])}")
    return "\n\n".join(sections) + "\n"
