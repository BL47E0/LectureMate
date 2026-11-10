"""LectureMate - turn a lecture recording into study material, 100% locally.

Run with:  streamlit run app.py
"""

import os

# Windows fixes: avoid a harmless-but-fatal "OMP: Error #15" and a noisy Hugging Face warning.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import shutil
import time
import traceback
from pathlib import Path

import streamlit as st

from modules import audio_processor, pdf_reader, utils
from modules.llm import DEFAULT_MODEL, LLMError, OllamaClient
from modules.study_generator import StudyGenerator, to_markdown
from modules.transcriber import transcribe

st.set_page_config(page_title="LectureMate", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .lm-title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0;}
      .lm-sub {color: #6b7280; font-size: 1.1rem; margin-top: 0.2rem; margin-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

WHISPER_CHOICES = {
    "base - recommended (fast, good)": "base",
    "tiny - fastest, least accurate": "tiny",
    "small - slower, more accurate": "small",
}
LANGUAGE_CHOICES = {"English": "en", "Auto-detect": None}
GENERATE_ORDER = [
    ("summary", "Summary"),
    ("notes", "Notes"),
    ("concepts", "Key concepts"),
    ("mcqs", "Quiz"),
    ("flashcards", "Flashcards"),
]


# ------------------------------------------------------------------ state
def init_state() -> None:
    defaults = {"lecture": None, "ctx": None, "results": {}, "chat": [], "flash": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_lecture() -> None:
    st.session_state.lecture = None
    st.session_state.ctx = None
    st.session_state.results = {}
    st.session_state.chat = []


def get_generator() -> StudyGenerator:
    model = st.session_state.get("llm_model") or DEFAULT_MODEL
    return StudyGenerator(OllamaClient(model=model))


# ------------------------------------------------------------- processing
def process_lecture(upload, slides_file, whisper_size: str, language) -> bool:
    """Upload -> (FFmpeg) -> Whisper -> transcript. Returns True on success."""
    utils.ensure_dirs()
    saved_path = None
    audio_path = None
    status = st.status("Processing lecture...", expanded=True)
    try:
        utils.validate_upload(upload.name, upload.size)

        status.write("💾 Saving the upload...")
        saved_path = utils.UPLOAD_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{utils.safe_filename(upload.name)}"
        upload.seek(0)
        with open(saved_path, "wb") as out:
            shutil.copyfileobj(upload, out)
        if saved_path.stat().st_size == 0:
            raise utils.LectureMateError("The uploaded file is empty.")

        if utils.file_kind(upload.name) == "video":
            status.write("🎞️ Extracting the audio track with FFmpeg...")
        audio_path, note = audio_processor.prepare_audio(saved_path, utils.TEMP_DIR)
        if note:
            status.write(f"⚠️ {note}")

        status.write(f"📝 Transcribing with Whisper ({whisper_size}). This runs on your CPU and can take a few minutes...")
        bar = status.progress(0.0, text="Starting transcription...")

        def on_progress(fraction: float) -> None:
            bar.progress(fraction, text=f"Transcribing... {int(fraction * 100)}%")

        result = transcribe(audio_path, whisper_size, language, on_progress)
        bar.progress(1.0, text="Transcription complete")

        slides_text = ""
        if slides_file is not None:
            status.write("📄 Reading the slides PDF...")
            try:
                slides_text = pdf_reader.extract_pdf_text(slides_file)
            except utils.LectureMateError as exc:
                status.write(f"⚠️ {exc}")

        stem = Path(utils.safe_filename(upload.name)).stem
        transcript_file = utils.OUTPUT_DIR / f"{stem}_transcript.txt"
        utils.save_text(transcript_file, result.text)

        reset_lecture()
        st.session_state.lecture = {
            "name": upload.name,
            "stem": stem,
            "kind": utils.file_kind(upload.name),
            "size": upload.size,
            "transcript": result.text,
            "timestamped": result.timestamped,
            "duration": result.duration,
            "language": result.language,
            "slides_text": slides_text,
            "slides_name": slides_file.name if slides_text and slides_file else "",
        }
        status.update(label="Lecture processed", state="complete", expanded=False)
        st.session_state.flash = f"Transcript ready ({len(result.text.split()):,} words). Now generate study material."
        return True
    except utils.LectureMateError as exc:
        status.update(label="Processing failed", state="error")
        st.error(str(exc))
    except Exception as exc:  # never crash the app on an unexpected error
        status.update(label="Processing failed", state="error")
        st.error(f"Something unexpected went wrong: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
    finally:
        utils.remove_files(saved_path, audio_path)  # temporary copies are not needed any more
    return False


# ------------------------------------------------------------- generation
def ensure_context(gen: StudyGenerator):
    """Prepare the transcript for the LLM once (condenses very long lectures)."""
    if st.session_state.ctx is not None:
        return st.session_state.ctx
    lecture = st.session_state.lecture
    holder = {}

    def on_progress(fraction: float, message: str) -> None:
        if "bar" not in holder:
            holder["bar"] = st.progress(0.0, text=message)
        holder["bar"].progress(min(fraction, 1.0), text=message)

    ctx = gen.build_context(lecture["transcript"], lecture["slides_text"], on_progress)
    if "bar" in holder:
        holder["bar"].empty()
    st.session_state.ctx = ctx
    return ctx


def generate(kind: str, **kwargs) -> bool:
    """Generate one kind of study material and store it. Shows errors in the UI."""
    gen = get_generator()
    try:
        with st.spinner("The local LLM is working... (on a CPU this can take a minute or two)"):
            ctx = ensure_context(gen)
            result = getattr(gen, kind)(ctx, **kwargs)
    except LLMError as exc:
        st.error(str(exc))
        return False
    except Exception as exc:
        st.error(f"Something unexpected went wrong: {exc}")
        return False
    if isinstance(result, dict):
        result["uid"] = time.time_ns()  # lets quiz widgets reset when a new quiz is made
    st.session_state.results[kind] = result
    return True


def generate_everything() -> None:
    with st.status("Generating study material...", expanded=True) as status:
        for kind, label in GENERATE_ORDER:
            status.write(f"⏳ {label}...")
            if not generate(kind):
                status.update(label=f"Stopped at: {label}", state="error")
                return
        status.update(label="All study material generated", state="complete", expanded=False)


# ------------------------------------------------------------------- tabs
def need_lecture() -> None:
    st.info("👈 Upload a lecture in the sidebar and click **Process lecture** first.")


def render_transcript_tab(lecture) -> None:
    minutes = lecture["duration"] / 60
    words = len(lecture["transcript"].split())
    cols = st.columns(4)
    cols[0].metric("Length", f"{minutes:.1f} min")
    cols[1].metric("Words", f"{words:,}")
    cols[2].metric("Language", lecture["language"])
    cols[3].metric("Slides PDF", "included" if lecture["slides_text"] else "none")

    show_times = st.checkbox("Show timestamps", value=False)
    with st.container(height=420, border=True):
        st.text(lecture["timestamped"] if show_times else lecture["transcript"])
    st.download_button(
        "⬇ Download transcript (.txt)",
        data=lecture["transcript"],
        file_name=f"{lecture['stem']}_transcript.txt",
        mime="text/plain",
    )


def render_text_tab(kind: str, button_label: str, hint: str) -> None:
    if st.button(button_label, key=f"btn_{kind}"):
        generate(kind)
    result = st.session_state.results.get(kind)
    if result:
        st.markdown(result)
    else:
        st.info(hint)


def render_quiz_tab() -> None:
    count = st.number_input("Number of questions", min_value=3, max_value=20, value=10, step=1)
    if st.button("Generate quiz", key="btn_mcqs"):
        generate("mcqs", n=int(count))

    result = st.session_state.results.get("mcqs")
    if not result:
        st.info("Click **Generate quiz** to create multiple-choice questions from the lecture.")
        return
    if not result["items"]:
        st.warning("The model's answer could not be turned into a quiz. Showing its raw output - try generating again.")
        st.text(result["raw"] or "(empty)")
        return

    show_all = st.toggle("Show all answers", value=False)
    for i, q in enumerate(result["items"], start=1):
        st.markdown(f"**Question {i}.** {q['question']}")
        labels = [f"{'ABCD'[j]}. {opt}" for j, opt in enumerate(q["options"])]
        choice = st.radio(
            f"Question {i} options", labels, index=None,
            key=f"mcq_{result['uid']}_{i}", label_visibility="collapsed",
        )
        correct_text = labels["ABCD".index(q["correct"])]
        if choice is not None:
            if choice[0] == q["correct"]:
                st.success(f"✅ Correct! {q['explanation']}")
            else:
                st.error(f"❌ Not quite. Correct answer: {correct_text}. {q['explanation']}")
        elif show_all:
            st.info(f"Correct answer: {correct_text}\n\n{q['explanation']}")
        st.divider()


def render_flashcards_tab() -> None:
    count = st.number_input("Number of flashcards", min_value=3, max_value=20, value=10, step=1)
    if st.button("Generate flashcards", key="btn_flashcards"):
        generate("flashcards", n=int(count))

    result = st.session_state.results.get("flashcards")
    if not result:
        st.info("Click **Generate flashcards** to create question/answer cards.")
        return
    if not result["items"]:
        st.warning("The model's answer could not be turned into flashcards. Showing its raw output - try again.")
        st.text(result["raw"] or "(empty)")
        return
    st.caption("Click a card to reveal the answer.")
    for i, card in enumerate(result["items"], start=1):
        with st.expander(f"Card {i}: {card['question']}"):
            st.markdown(f"**Answer:** {card['answer']}")


def render_ask_tab() -> None:
    st.caption("Answers come only from your lecture. If the lecture doesn't cover it, the assistant will say so.")
    examples = [
        "What is the main topic of this lecture?",
        "What examples did the lecturer give?",
        "What are the most important topics for an exam?",
        "Give me possible 10-mark questions.",
    ]
    cols = st.columns(len(examples))
    pressed = [ex for col, ex in zip(cols, examples) if col.button(ex, key=f"ex_{ex}")]
    clicked = pressed[0] if pressed else None

    chat_box = st.container()
    with chat_box:
        for role, content in st.session_state.chat:
            with st.chat_message(role):
                st.markdown(content)

    with st.form("ask_form", clear_on_submit=True):
        typed = st.text_input("Ask a question about the lecture", placeholder="e.g. Explain process scheduling in simple words")
        submitted = st.form_submit_button("Ask")
    question = clicked or (typed.strip() if submitted else "")
    if not question:
        return

    history = [{"role": r, "content": c} for r, c in st.session_state.chat[-6:]]
    gen = get_generator()
    with chat_box:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                ctx = ensure_context(gen)
                answer = st.write_stream(gen.answer_stream(ctx, question, history))
            except LLMError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Something unexpected went wrong: {exc}")
                return
    st.session_state.chat.append(("user", question))
    st.session_state.chat.append(("assistant", answer))


# ------------------------------------------------------------------- main
init_state()

with st.sidebar:
    st.title("📚 LectureMate")

    st.subheader("1. Upload lecture")
    upload = st.file_uploader(
        "Lecture recording (MP3, WAV, M4A, MP4)",
        type=[e.lstrip(".") for e in sorted(utils.SUPPORTED_EXTENSIONS)],
    )
    if upload is not None:
        st.markdown(
            f"**File:** {upload.name}  \n"
            f"**Type:** {Path(upload.name).suffix.upper().lstrip('.')} ({utils.file_kind(upload.name)})  \n"
            f"**Size:** {utils.human_size(upload.size)}"
        )
    slides_file = st.file_uploader("Lecture slides (optional PDF)", type=["pdf"])

    with st.expander("⚙️ Settings"):
        whisper_label = st.selectbox("Speech model (Whisper)", list(WHISPER_CHOICES))
        language_label = st.selectbox("Lecture language", list(LANGUAGE_CHOICES))
        st.text_input("Ollama model", value=DEFAULT_MODEL, key="llm_model", help="Must be downloaded with `ollama pull <name>`.")

    process_clicked = st.button("▶ Process lecture", type="primary", disabled=upload is None)
    if st.session_state.lecture is not None and st.button("🗑 Clear lecture"):
        reset_lecture()
        st.rerun()

    st.subheader("2. Generate study material")
    run_all = st.button("✨ Generate everything", disabled=st.session_state.lecture is None)

    st.subheader("System check")
    health = OllamaClient(model=st.session_state.get("llm_model") or DEFAULT_MODEL).status()
    if not health["running"]:
        st.error("Ollama: not running")
        st.caption(health["message"])
    elif not health["model_ready"]:
        st.warning("Ollama: running, model missing")
        st.caption(health["message"])
    else:
        st.success("Ollama + model: ready")
    if audio_processor.find_ffmpeg():
        st.success("FFmpeg: found")
    else:
        st.warning("FFmpeg: not found (needed for MP4)")

st.markdown('<p class="lm-title">🎓 LectureMate</p>', unsafe_allow_html=True)
st.markdown('<p class="lm-sub">Turn your lectures into intelligent study material.</p>', unsafe_allow_html=True)

if process_clicked and upload is not None:
    if process_lecture(upload, slides_file, WHISPER_CHOICES[whisper_label], LANGUAGE_CHOICES[language_label]):
        st.rerun()  # refresh so the sidebar buttons become enabled

if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

if run_all and st.session_state.lecture is not None:
    generate_everything()

lecture = st.session_state.lecture
tabs = st.tabs(["📄 Transcript", "📝 Summary", "📒 Notes", "🔑 Key Concepts", "❓ Quiz", "🃏 Flashcards", "💬 Ask Lecture"])

with tabs[0]:
    if lecture:
        render_transcript_tab(lecture)
    else:
        need_lecture()
with tabs[1]:
    if lecture:
        render_text_tab("summary", "Generate summary", "Click **Generate summary** for a short overview of the lecture.")
    else:
        need_lecture()
with tabs[2]:
    if lecture:
        render_text_tab("notes", "Generate notes", "Click **Generate notes** for structured study notes.")
    else:
        need_lecture()
with tabs[3]:
    if lecture:
        render_text_tab("concepts", "Extract key concepts", "Click **Extract key concepts** to list the main ideas.")
    else:
        need_lecture()
with tabs[4]:
    if lecture:
        render_quiz_tab()
    else:
        need_lecture()
with tabs[5]:
    if lecture:
        render_flashcards_tab()
    else:
        need_lecture()
with tabs[6]:
    if lecture:
        render_ask_tab()
    else:
        need_lecture()

# Rendered after the tabs so it always includes what was just generated.
if lecture and st.session_state.results:
    st.divider()
    st.download_button(
        "⬇ Download all study material (.md)",
        data=to_markdown(lecture["name"], st.session_state.results),
        file_name=f"{lecture['stem']}_study_material.md",
        mime="text/markdown",
    )
