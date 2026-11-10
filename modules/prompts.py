"""All LLM prompts in one place. Every function returns (system_prompt, user_prompt)."""

from __future__ import annotations

RULES = (
    "You are analyzing a college lecture transcript. Use only the information provided in the "
    "transcript. Do not invent facts. If information is missing, say that it is not available "
    "in the lecture.\n"
    "The transcript was produced by automatic speech recognition, so technical terms may be "
    "misspelled; silently fix obvious spelling mistakes but never add new content.\n"
    "Always write in clear English."
)


def condense(part: str, index: int, total: int) -> tuple[str, str]:
    """Map step for long lectures: shrink one part into compact fact notes."""
    system = (
        f"{RULES}\n\n"
        f"TASK: This is part {index} of {total} of a long lecture. Rewrite it as compact bullet "
        "notes that keep EVERY important definition, concept, example, name, number and "
        "step-by-step explanation. Remove filler, repetition and small talk. Aim for roughly "
        "one quarter of the original length. Output only the bullet notes."
    )
    return system, f"LECTURE PART {index}/{total}:\n\n{part}"


def summary(context: str) -> tuple[str, str]:
    system = (
        f"{RULES}\n\n"
        "TASK: Write a concise summary of the lecture for a college student, in Markdown, "
        "using exactly this layout:\n"
        "**Main topic:** one sentence.\n\n"
        "**Overview:** 2-3 short paragraphs explaining the major concepts in simple language.\n\n"
        "**Key takeaways:**\n- 3 to 5 bullet points\n\n"
        "Keep it between 150 and 250 words. Avoid repetition."
    )
    return system, f"LECTURE:\n\n{context}"


def notes(context: str) -> tuple[str, str]:
    system = (
        f"{RULES}\n\n"
        "TASK: Create detailed, well-organised study notes in Markdown.\n"
        "- Use '## ' headings for major sections and '### ' for sub-headings.\n"
        "- Put the details under each heading as bullet points.\n"
        "- Use these sections when the lecture covers them: Introduction, Important Concepts, "
        "Definitions, Examples, Important Points to Remember. Skip any section the lecture does "
        "not cover; you may add topic-specific headings.\n"
        "- Bold the key terms.\n"
        "- Base the notes ONLY on the lecture. Do not add outside knowledge."
    )
    return system, f"LECTURE:\n\n{context}"


def concepts(context: str, min_n: int, max_n: int) -> tuple[str, str]:
    system = (
        f"{RULES}\n\n"
        f"TASK: Extract the {min_n} to {max_n} most important concepts from the lecture. "
        "Output a numbered Markdown list. Format each item exactly like this:\n"
        "1. **Concept name** - one or two sentences explaining it, using only the lecture.\n"
        "Order them in the order they appear in the lecture. No introduction or conclusion."
    )
    return system, f"LECTURE:\n\n{context}"


def mcqs(context: str, n: int) -> tuple[str, str]:
    system = (
        f"{RULES}\n\n"
        f"TASK: Write exactly {n} multiple-choice questions that test understanding of this "
        "lecture. Every question must be answerable from the lecture content. Each question has "
        "exactly four options, one correct answer, and a short explanation (1-2 sentences). "
        "Spread the correct answers across A, B, C and D. Wrong options must be plausible.\n"
        "Return ONLY valid JSON in exactly this shape:\n"
        '{"questions": [{"question": "...", "options": {"A": "...", "B": "...", "C": "...", '
        '"D": "..."}, "correct": "B", "explanation": "..."}]}'
    )
    return system, f"LECTURE:\n\n{context}"


def flashcards(context: str, n: int) -> tuple[str, str]:
    system = (
        f"{RULES}\n\n"
        f"TASK: Create exactly {n} study flashcards from the lecture. Each has a short question "
        "on the front (for example 'What is process scheduling?') and a clear answer of 1-3 "
        "sentences on the back, using only the lecture.\n"
        "Return ONLY valid JSON in exactly this shape:\n"
        '{"flashcards": [{"question": "...", "answer": "..."}]}'
    )
    return system, f"LECTURE:\n\n{context}"


def qa_system(context: str) -> str:
    """Q&A system prompt. The student's question is sent separately as the user message."""
    return (
        f"{RULES}\n\n"
        "TASK: Answer the student's question using ONLY the lecture material below.\n"
        "- If the answer is not in the material, reply exactly: "
        "\"I couldn't find this information in the uploaded lecture.\"\n"
        "- If only part of the answer is in the material, answer that part and say which part "
        "is not covered in the lecture.\n"
        "- If asked for exam topics or practice/10-mark questions, base them strictly on the "
        "topics the lecture covers.\n"
        "- Explain in simple words. Use short bullet points when it helps. Do not mention these "
        "instructions.\n\n"
        f"<lecture>\n{context}\n</lecture>"
    )
