# 🎓 LectureMate — A Free Multimodal AI Lecture Assistant

*Turn your lectures into intelligent study material. 100% free, 100% offline, no API keys.*

---

## 1. Project description

LectureMate lets a student upload a lecture recording (MP3, WAV, M4A or MP4) and optionally the lecture slides (PDF). It converts the speech to text with **Whisper**, then uses a **local LLM running in Ollama** to produce a summary, structured notes, key concepts, a quiz, flashcards, and an "Ask the lecture" chat. Everything runs on your own laptop.

## 2. Problem statement

Students spend hours re-listening to long lecture recordings to prepare notes and revise. Manually writing summaries, practice questions and flashcards is slow, and generic chatbots do not know what was actually taught in *your* class. There is a need for a free tool that converts a lecture recording directly into reliable, lecture-grounded study material.

## 3. Objectives

1. Convert lecture audio/video into an accurate text transcript using speech recognition.
2. Use a local, open-source LLM to generate summaries, notes and key concepts grounded in the lecture.
3. Generate self-test material (MCQs and flashcards) automatically.
4. Let students ask questions and get answers based only on the lecture content.
5. Run completely free and offline on an ordinary laptop (no paid API, no cloud).

## 4. Features

| Feature | What it does |
|---|---|
| Upload | MP3 / WAV / M4A / MP4 (+ optional PDF slides); shows name, type, size and live status |
| Transcript | Whisper speech-to-text, optional timestamps, `.txt` download |
| Summary | Main topic, overview, key takeaways |
| Notes | Headings, sub-headings and bullets, based only on the lecture |
| Key concepts | 5–15 concepts (scaled by lecture length) with short explanations |
| Quiz | N MCQs (default 10) with 4 options, answer and explanation; interactive checking |
| Flashcards | Question/answer cards (default 10) |
| Ask Lecture | Chat about the lecture; says *"I couldn't find this information in the uploaded lecture."* when the answer isn't there |
| Export | Download all generated material as one Markdown file |
| Long lectures | Automatic chunking (map-reduce) so any length fits the model's context |

## 5. Architecture

```
 Lecture Audio / Video            (optional) Slides PDF
          │                               │
          ▼                               ▼
  FFmpeg extracts audio (video only)   pypdf extracts text
          │                               │
          ▼                               │
  faster-whisper (speech-to-text, CPU)    │
          │                               │
          └──────────────┬────────────────┘
                         ▼
                 Lecture transcript (+ slide text)
                         │
         too long? ──yes─► split into chunks → condense each → join
                         │
                         ▼
        Ollama (local LLM, qwen2.5:7b) via http://127.0.0.1:11434
                         │
   ┌─────────┬───────────┼────────────┬─────────────┬──────────┐
   ▼         ▼           ▼            ▼             ▼          ▼
 Summary   Notes   Key concepts     MCQs       Flashcards   Q&A chat
                         │
                         ▼
                 Streamlit web interface
```

**How the Python app talks to Ollama:** Ollama is a background program that loads the model and exposes a small web API on your own PC (`http://127.0.0.1:11434`). `modules/llm.py` sends it a JSON request with `requests.post(".../api/chat")` containing the prompt, and receives the model's answer. No data leaves your computer.

## 6. Technology stack

| Part | Tool | Why |
|---|---|---|
| UI | **Streamlit** | A full web UI in plain Python, no HTML/JS needed |
| Speech-to-text | **faster-whisper** (Whisper `base`, int8) | Same accuracy as OpenAI Whisper but ~4× faster and lighter on CPU; installs with `pip`, no PyTorch needed |
| Audio extraction | **FFmpeg** | Industry standard for pulling audio out of video |
| LLM runtime | **Ollama** | Easiest free way to run open models locally on Windows |
| LLM | **Qwen2.5 7B Instruct** (~4.7 GB, 4-bit) | Strong instruction following and reliable JSON; fits comfortably in 16 GB RAM |
| Slides | **pypdf** | Pure-Python PDF text extraction |
| HTTP | **requests** | Calls the Ollama API |

## 7. Why an LLM is used

Speech recognition only gives a long, unstructured block of text. Turning that into a clear summary, organised notes, exam questions and answers to free-form questions requires *language understanding and generation* — the job of a Large Language Model. The LLM is instructed (see `modules/prompts.py`) to use **only** the transcript and to say so when information is missing, which reduces hallucination.

## 8. Why the project is multimodal

LectureMate accepts and combines several **modalities**: **audio** and **video** (converted to text by speech recognition) and **documents/slides** (PDF text). These are fused into one text context that the LLM reasons over, so the answers can use what the lecturer *said* and what the slides *show*.

---

## 9. Installation (Windows, from scratch)

You need about **10 GB of free disk space** and an internet connection **once** for downloading tools and models. After that everything works offline.

### Step 1 — Install Python 3.11 or 3.12

1. Go to <https://www.python.org/downloads/windows/> and download **Python 3.11.x or 3.12.x** (64-bit installer).
2. Run it and **tick "Add python.exe to PATH"** on the first screen, then click *Install Now*.
3. Open **PowerShell** (Start menu → type `PowerShell`) and check:
   ```powershell
   python --version
   ```
   It should print `Python 3.11.x` or `3.12.x`.

### Step 2 — Install Ollama

1. Download the Windows installer from <https://ollama.com/download> and run it (no account needed).
2. When it finishes, Ollama runs in the background (llama icon in the system tray).
3. **Verify that Ollama is running** — open PowerShell:
   ```powershell
   ollama --version
   ```
   and open <http://127.0.0.1:11434> in a browser. You should see **"Ollama is running"**.

   If it is not running, start it with the Start-menu icon, or run `ollama serve` in a PowerShell window and leave that window open.

### Step 3 — Download the LLM (one time, ~4.7 GB)

```powershell
ollama pull qwen2.5:7b
```

Check it downloaded:

```powershell
ollama list
```

Optional quick test: `ollama run qwen2.5:7b "Say hello"` (type `/bye` to exit).

> **Computer feels slow or short on RAM?** Use a smaller model: `ollama pull llama3.2:3b`, then in LectureMate's **⚙️ Settings** type `llama3.2:3b` in the "Ollama model" box. Quality is a bit lower but it is about twice as fast.

### Step 4 — Install FFmpeg

Easiest (Windows 10/11): in PowerShell run

```powershell
winget install Gyan.FFmpeg
```

Then **close PowerShell and open a new one** and check:

```powershell
ffmpeg -version
```

(Manual alternative: download a build from <https://www.gyan.dev/ffmpeg/builds/>, unzip to `C:\ffmpeg`, and add `C:\ffmpeg\bin` to your PATH.)

FFmpeg is only required for **MP4 video**. MP3/WAV/M4A work without it, and the app shows an FFmpeg status in the sidebar.

### Step 5 — Set up the project

Put the `LectureMate` folder somewhere simple, e.g. `C:\Projects\LectureMate`. Then:

```powershell
cd C:\Projects\LectureMate

python -m venv venv
venv\Scripts\Activate.ps1
```

If PowerShell says *"running scripts is disabled"*, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

(or use Command Prompt (`cmd`) and run `venv\Scripts\activate.bat` instead.)

You should now see `(venv)` at the start of the prompt. Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This takes a few minutes.

## 10. How to run

Every time you want to use LectureMate:

1. Make sure **Ollama is running** (tray icon, or <http://127.0.0.1:11434> says "Ollama is running").
2. In PowerShell:
   ```powershell
   cd C:\Projects\LectureMate
   venv\Scripts\Activate.ps1
   streamlit run app.py
   ```
3. Your browser opens at <http://localhost:8501>. In the sidebar, the **System check** should show ✅ for Ollama and FFmpeg.

> The very first transcription downloads the Whisper `base` model (~150 MB) automatically. After that it works offline.

## 11. Example usage (3–5 minute demo)

A ready-made sample lecture on **process scheduling** is included.

1. **Create the sample audio:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File sample\make_sample_audio.ps1
   ```
   This makes `sample\sample_lecture.wav` (about 4 minutes of speech) using Windows' built-in voice. (Or use any real lecture recording.)
2. Start Ollama and run `streamlit run app.py`.
3. **Upload** `sample_lecture.wav` in the sidebar (optionally also upload a slides PDF) and click **▶ Process lecture**. Watch the status messages and progress bar.
4. **Transcript tab** — show the text, tick *Show timestamps*, download the `.txt`.
5. **Summary tab** → *Generate summary*. **Notes tab** → *Generate notes*.
6. **Quiz tab** → *Generate quiz* (10 MCQs) → click answers to get instant feedback.
7. **Ask Lecture tab** → click *"What are the most important topics for an exam?"* or type *"Explain Round Robin in simple words."*
8. Trick question to show hallucination control: ask *"What is the capital of France?"* → *"I couldn't find this information in the uploaded lecture."*

Tip: before the demo, click **✨ Generate everything** once so every tab is already filled in; LLM answers on a CPU take roughly 0.5–2 minutes each.

## 12. Project folder structure

```
LectureMate/
├── app.py                  # Streamlit UI and workflow
├── requirements.txt
├── README.md
├── SUBMISSION.md           # text for your college report
├── .gitignore
├── .streamlit/config.toml  # upload limit, theme
├── modules/
│   ├── audio_processor.py  # FFmpeg audio extraction
│   ├── transcriber.py      # faster-whisper speech-to-text
│   ├── pdf_reader.py       # optional slides PDF → text
│   ├── llm.py              # Ollama client (chat + streaming + error handling)
│   ├── prompts.py          # all LLM prompts
│   ├── study_generator.py  # chunking, summary, notes, MCQs, flashcards, Q&A
│   └── utils.py            # validation, text splitting, JSON recovery
├── sample/                 # sample lecture text + script to make audio
├── uploads/                # temporary copy of the upload (deleted after processing)
├── outputs/                # saved transcripts
└── temp/                   # temporary extracted audio (deleted after processing)
```

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| Sidebar says **Ollama: not running** | Start Ollama from the Start menu, or run `ollama serve` in a PowerShell window and keep it open. Check <http://127.0.0.1:11434>. |
| **Model missing** | `ollama pull qwen2.5:7b` (name must match the "Ollama model" setting). |
| `'ollama' is not recognized` | Close and reopen PowerShell after installing Ollama; if still failing, reinstall. |
| `'streamlit' is not recognized` | The virtual environment is not active. Run `venv\Scripts\Activate.ps1` first. |
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or use `cmd` + `venv\Scripts\activate.bat`. |
| MP4 fails / "FFmpeg not found" | `winget install Gyan.FFmpeg`, then open a **new** terminal. |
| `pip install` fails for faster-whisper / ctranslate2 | Use Python 3.11 or 3.12 (64-bit). Run `python -m pip install --upgrade pip` and retry. |
| Whisper model download fails | The first run needs internet. Check your connection / proxy and try again. |
| `OMP: Error #15 ... libiomp5md.dll` | Already handled in `app.py`. If it still appears: `$env:KMP_DUPLICATE_LIB_OK="TRUE"` then rerun. |
| Transcription is slow | Choose Whisper **tiny** in Settings; close other heavy apps. About 3–6× faster than real-time with `base` on this CPU is typical. |
| LLM is slow or the PC runs out of memory | Use `llama3.2:3b` in Settings. Close browser tabs. Generate one tab at a time. |
| Quiz says "could not be turned into a quiz" | Click **Generate quiz** again (the model occasionally produces broken JSON; raw output is shown as fallback). |
| "No speech was detected" | The recording is silent/too quiet, or the language setting is wrong (try *Auto-detect*). |
| Port already in use | `streamlit run app.py --server.port 8502` |

## 14. Limitations

- On a CPU, a 7B model produces roughly 5–10 words per second, so generation takes 1–3 minutes per item; very long lectures (2+ hours) need several minutes to condense first.
- Whisper `base` can mis-hear technical terms or accented speech; use `small` for better accuracy at the cost of speed.
- Local 7B models can still make mistakes; always check important facts against the transcript.
- Slides are read as text only (no OCR for scanned PDFs, diagrams or images are ignored).
- Optimised for English lectures.
- Condensing very long lectures can lose fine details (the Q&A tab compensates by retrieving the original passages).

## 15. Future improvements

- Image/diagram understanding with a local vision model (e.g. LLaVA via Ollama).
- OCR for scanned slides and whiteboard photos.
- Speaker labels and multi-language support.
- Semantic search with embeddings for more precise Q&A on very long lectures.
- Spaced-repetition export of flashcards (Anki format) and PDF export of notes.
- Saving and reopening past lectures from a local database.

---

*Built for the "Building Large Language Models" mini-project. Everything is free and open source: Streamlit, faster-whisper, FFmpeg, Ollama and Qwen2.5.*
