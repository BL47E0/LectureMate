# LectureMate — College Submission Content

Copy the sections below into your report / PPT.

## Title

**LectureMate — A Multimodal AI Lecture Assistant**

## Problem Statement

Students rely heavily on recorded lectures, but converting an hour of speech into usable study material (notes, summaries, practice questions) is slow and manual. Existing chatbots do not know the content of a specific lecture and often hallucinate, while cloud-based tools require paid subscriptions or send private data to external servers. The problem is to design a free, privacy-preserving system that converts lecture audio/video (and slides) into accurate, lecture-grounded study material and allows students to question the lecture interactively.

## Objectives

1. To convert lecture audio and video recordings into text using automatic speech recognition (Whisper).
2. To use a locally hosted Large Language Model to generate summaries, structured notes and key concepts from the transcript.
3. To automatically create multiple-choice questions and flashcards for self-assessment.
4. To provide a question-answering interface that answers strictly from the lecture content and reports when information is unavailable.
5. To handle long lectures efficiently on a laptop using transcript chunking.
6. To build the entire system using free, open-source tools that run offline on consumer hardware.

## Proposed Solution

LectureMate is a Streamlit web application that runs entirely on the student's laptop. The student uploads a lecture recording (MP3, WAV, M4A or MP4) and optionally the lecture slides as a PDF. The system extracts the audio, transcribes it with faster-whisper, and passes the transcript to a local LLM (Qwen2.5 7B served by Ollama). Carefully designed prompts instruct the model to use only the lecture content. The application then presents the transcript, summary, notes, key concepts, a quiz, flashcards and a chat interface in separate tabs.

## Methodology

```
Audio / Video  (+ optional slides PDF)
      ↓
Audio extraction (FFmpeg, video only)
      ↓
Speech Recognition (faster-whisper, CPU, int8)
      ↓
Transcript  (+ text from slides)
      ↓
Chunking & condensing if the transcript exceeds the model's context window
      ↓
Local LLM (Qwen2.5 7B via Ollama) with task-specific, grounded prompts
      ↓
Study Material: Summary · Notes · Key Concepts · MCQs · Flashcards · Q&A
```

1. **Input handling:** file type, size and emptiness are validated.
2. **Speech-to-text:** audio is resampled and transcribed using Whisper with voice-activity detection.
3. **Context preparation:** short transcripts are used directly; long ones are split into chunks, each chunk is condensed by the LLM, and the notes are merged (map-reduce).
4. **Generation:** separate prompts for each output type; MCQs and flashcards are requested as JSON and parsed with a fault-tolerant parser, with a raw-text fallback.
5. **Grounded Q&A:** answers use only the lecture context (for very long lectures, the most relevant original passages are retrieved by keyword matching); if the answer is absent the assistant replies that it could not find it.

## Technologies Used

- Python 3.11
- Streamlit (web interface)
- faster-whisper / OpenAI Whisper model (speech recognition)
- FFmpeg (audio extraction)
- Ollama (local LLM runtime)
- Qwen2.5 7B Instruct, 4-bit quantised (Large Language Model)
- pypdf (slide text extraction)
- requests (communication with the Ollama REST API)

## Expected Outcome

For any uploaded lecture the system produces: (1) a downloadable transcript, (2) a concise summary, (3) structured study notes, (4) a list of key concepts with explanations, (5) a 10-question MCQ quiz with answers and explanations, (6) flashcards, and (7) an interactive Q&A assistant that answers only from the lecture. This reduces the time needed to prepare revision material from hours to a few minutes, at zero cost and with complete data privacy.

## Future Scope

1. Understanding slide images and diagrams with a local vision-language model.
2. OCR support for scanned slides and handwritten whiteboard photos.
3. Multilingual lectures (e.g. Hindi, Malayalam) with translation.
4. Embedding-based retrieval (RAG) for precise answers over a whole semester of lectures.
5. Export to Anki flashcards and PDF/Word notes.
6. Live (real-time) lecture transcription and note-taking, with a database to store lecture history.
