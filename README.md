# MarkItDown App — Document → Markdown converter (UI by Kshitij)

A Streamlit desktop-style web app that turns **documents, images, audio, video, subtitles, presentations, spreadsheets, e-mails, ZIP archives and web/YouTube URLs into clean Markdown** — and then exports that Markdown as `.md`, `.txt`, `.json`, `.html`, `.docx` or `.pdf`.

It wraps Microsoft's [MarkItDown](https://github.com/microsoft/markitdown) and adds four optional engines that run **fully offline on your own machine**:

| Engine | What it does | Needs |
|---|---|---|
| **Tesseract OCR** | Free text extraction from images | Tesseract installed |
| **Anthropic Claude** | Structures / summarises documents, describes images | API key |
| **Ollama (Llama & friends)** | Same as Claude but local and free; text *and* vision models | Ollama installed |
| **Unlimited-OCR** (Baidu) | State-of-the-art document parsing of scanned PDFs & images — tables, layout, bounding boxes | NVIDIA GPU ≥ 8 GB |
| **faster-whisper** | Transcribes audio/video with timestamps (CUDA, CPU fallback) | (GPU optional) |

Everything is driven by a single double-clickable launcher, `Run MarkItDown.bat`, which creates the Python environment, installs dependencies, picks a free port and opens your browser.

---

## Contents

1. [Features at a glance](#1-features-at-a-glance)
2. [Requirements](#2-requirements)
3. [Installation (Windows, one click)](#3-installation-windows-one-click)
4. [Running the app](#4-running-the-app)
5. [How a session runs — walkthrough](#5-how-a-session-runs--walkthrough)
6. [Supported inputs](#6-supported-inputs)
7. [Conversion engines & settings](#7-conversion-engines--settings)
8. [Outputs](#8-outputs)
9. [Project structure](#9-project-structure)
10. [Configuration](#10-configuration)
11. [Manual setup (no launcher / macOS / Linux)](#11-manual-setup-no-launcher--macos--linux)
12. [Troubleshooting](#12-troubleshooting)
13. [Credits](#13-credits)

---

## 1. Features at a glance

- **Batch upload** — drop many files at once; each converts with its own progress bar and stage messages.
- **URL conversion** — any web page (three fallbacks: MarkItDown → browser-header fetch → Playwright headless Chromium), Wikipedia articles, and **YouTube captions** (English → Hindi → Marathi priority, `**[MM:SS]**` timestamps). If a video has no captions, an inline uploader lets you drop the video file for Whisper transcription right there.
- **Side-by-side preview** — original on the left (PDFs are rasterised page-by-page with PyMuPDF so Chrome's iframe block doesn't matter; images, text, audio, video previews), Markdown on the right.
- **Editable output** — fix the Markdown in-app before downloading.
- **Copy to clipboard** button.
- **Multi-format download** — tick any of `.md .txt .json .html .docx .pdf`; several at once come as a ZIP.
- **JSON sidecar** — structured data alongside the Markdown (tables as arrays, subtitle cues, transcript segments, page counts, detected block types and raw bounding boxes for Unlimited-OCR).
- **History** — the last 10 conversions of the session are re-openable from the sidebar.
- **Model guidance** — every model picker shows a "What is each model best at?" note; the Ollama pickers list the models actually installed on your machine.
- **GPU warm-up with progress** — Unlimited-OCR loads once at startup with a staged progress bar (config → tokenizer → CPU weights → GPU → CUDA kernels → warm-up inference), then every later conversion is instant.
- **Smart launcher** — free-port picking (never collides with another Streamlit app), auto browser open, one-time dependency install with version markers, optional GPU extras prompt.

## 2. Requirements

### Mandatory
| | |
|---|---|
| OS | Windows 10/11 (launcher is a `.bat`; the Python code itself is cross-platform — see §11) |
| Python | **3.12** recommended (`py -3.12`); any `python` on PATH is used as fallback |
| Disk | ~1.5 GB for the base virtualenv + Playwright Chromium |
| Internet | first run only (package install, model downloads) |

### Optional — enable individual features
| Feature | Requirement | Where to get it |
|---|---|---|
| Free image OCR | Tesseract at `C:\Program Files\Tesseract-OCR\tesseract.exe` | https://github.com/UB-Mannheim/tesseract/wiki |
| Local AI (text + vision) | Ollama running on `localhost:11434` with some models pulled (`ollama pull llama3.2`, `ollama pull llama3.2-vision`, …) | https://ollama.com/download |
| Claude AI modes | Anthropic API key (entered in the sidebar, never stored) | https://console.anthropic.com |
| Unlimited-OCR + GPU Whisper | NVIDIA GPU with **≥ 8 GB VRAM**, driver supporting CUDA 12.8 (RTX 30/40/50-series). Downloads ~3 GB of PyTorch CUDA wheels + ~6.7 GB model weights on first use | installed by the launcher from `requirements-ocr.txt` |
| Whisper on CPU | nothing extra — `faster-whisper` falls back to CPU int8 automatically (slower) | — |

> ffmpeg is **not** required: video/audio tracks are decoded with PyAV, which ships with faster-whisper.

## 3. Installation (Windows, one click)

```powershell
git clone https://github.com/kshitijrm-ironman/MarkItDown-Converter.git
cd MarkItDown-Converter
```

Then **double-click `Run MarkItDown.bat`**. On the first run it will:

1. Locate Python 3.12 (or any Python).
2. Create `venv\` and install `requirements.txt` (MarkItDown, Streamlit, Playwright, handlers for every file type, output renderers). Takes a few minutes.
3. Install Chromium for Playwright (JavaScript-rendered pages).
4. If an NVIDIA GPU is detected, ask *"Install GPU extras now?"* (auto-skips after 10 s). Answer **Y** to install PyTorch CUDA 12.8, Unlimited-OCR's dependencies and faster-whisper from `requirements-ocr.txt`. You can do this later at any time:
   ```
   venv\Scripts\python -m pip install -r requirements-ocr.txt
   ```
5. Report whether Tesseract and Ollama were found (informational only).
6. Pick a free port, start Streamlit, wait until it answers, and open your browser.

Later runs skip straight to step 6 (a version marker in `venv\.deps_installed` re-installs packages only when `requirements.txt` changes).

## 4. Running the app

- **Start:** double-click `Run MarkItDown.bat` (or run it from a terminal to keep the log visible).
- **Stop:** press `Ctrl+C` in the launcher window, or close it.
- **Force a specific port:** `set STREAMLIT_SERVER_PORT=8502` before starting the launcher. Otherwise a random free port is chosen and written to `%TEMP%\markitdown_port.txt`.
- **Point at a remote Ollama:** `set OLLAMA_HOST=http://192.168.1.20:11434`.

The first time Unlimited-OCR is used (or at startup when the GPU extras are installed) the ~6.7 GB model is downloaded to the Hugging Face cache (`%USERPROFILE%\.cache\huggingface`). Whisper models (75 MB – 3 GB depending on size) are downloaded on first use as well.

## 5. How a session runs — walkthrough

**Launcher console (subsequent run, GPU machine):**

```
==================================================
  MarkItDown App - Document to Markdown converter
==================================================

[OK] Tesseract OCR found.
[OK] Ollama found.

[LAUNCH] Starting MarkItDown App... (Ctrl+C to stop)
[INFO] Serving on http://localhost:61994 (free port picked automatically)
[INFO] Your browser will open automatically once the server is up.

  You can now view your Streamlit app in your browser.
  URL: http://localhost:61994
```

**In the browser:**

1. **Startup** — if the GPU extras are installed, a progress bar runs once per session:
   `Loading model config… 0% → Loading tokenizer… 10% → Loading model weights onto CPU… 25% → Transferring weights to GPU… 60% → Compiling CUDA kernels… 80% → Running warm-up inference… 95% → Unlimited-OCR ready on GPU 100%`.
2. **Sidebar → Settings** — choose the *Document conversion method* (MarkItDown Default / Claude / Llama local / Unlimited-OCR). Image and audio/video settings appear automatically when such files are in the batch.
3. **Choose input method** — `📁 Upload File` or `🌐 Enter URL`.
4. **Upload one or more files** — a table lists name / detected type / size. Click **🔄 Convert to Markdown**.
5. **Watch progress** — each file gets a status box with staged percentages, e.g. for a scanned PDF:
   `Rasterizing PDF pages… 10% → Processing page 3 of 12… 29% → … → Assembling output… 90% → Generating output formats… 100%`;
   for a video: `Extracting audio track… 10% → Loading Whisper model… 20% → Transcribing 04:12 / 09:30 on CUDA — 57 segment(s) → Formatting transcript… 95% → Done`.
6. **Results** — a tab per file. Left: original preview (PDF page selector, image, video player…). Right: the Markdown in an editable text area with **📋 Copy to clipboard**.
7. **Download** — tick the formats you want (`.md` is pre-selected) and click **⬇️ Download**. Multiple formats arrive as one ZIP. Unlimited-OCR PDF conversions are additionally saved to `output\<name>.md` + `output\<name>.json`.
8. **History** — the sidebar's *Last 10 conversions* lets you re-open any earlier result of the session.

**YouTube example:** paste `https://www.youtube.com/watch?v=dQw4w9WgXcQ` → *YouTube video detected* → **Convert** → "✅ Transcript fetched — 61 caption cue(s), language: English" and the Markdown reads

```
**[00:18]** ♪ We're no strangers to love ♪

**[00:22]** ♪ You know the rules and so do I ♪
```

If the video has no captions you'll see *"No captions available for this video. Upload the video file directly for Whisper transcription."* with an uploader directly underneath.

## 6. Supported inputs

| Category | Extensions | How it is converted |
|---|---|---|
| Documents | `.pdf .docx .pptx .xlsx .html .htm .txt .md .json .xml …` (everything MarkItDown supports) | MarkItDown; scanned PDFs optionally via Unlimited-OCR; optional Claude/Ollama enhancement |
| Images | `.jpg .jpeg .png .bmp .gif .jfif .webp .heic .heif .tiff .tif` (multi-page TIFF handled page by page) | Tesseract OCR / Claude vision / Ollama vision / Unlimited-OCR |
| Audio | `.mp3 .wav .m4a .ogg .flac` | faster-whisper transcription with timestamps |
| Video | `.mp4 .mkv .avi .mov` | audio track decoded in-process → faster-whisper |
| Subtitles | `.srt .vtt` | direct parse to timestamped Markdown + JSON cues |
| Presentations | `.pptx` | python-pptx — slide titles, text, tables, speaker notes |
| Spreadsheets | `.xlsx .csv` | pandas → Markdown tables (per sheet), rows also in JSON |
| E-mail | `.eml` | stdlib `email` — headers, body, attachment list |
| Archives | `.zip` | extracted and every member converted (up to 50 files, 2 levels deep) |
| URLs | web pages, `wikipedia.org/wiki/…`, `youtube.com` / `youtu.be` | see §1 |

## 7. Conversion engines & settings

### Document conversion method (sidebar)
| Mode | Behaviour |
|---|---|
| ⚡ **MarkItDown Default** | Plain MarkItDown conversion. |
| 🤖 **Anthropic Claude AI** | MarkItDown first, then Claude restructures/summarises. Needs your API key. |
| 🦙 **Llama (Local, Free)** | Same, using an Ollama text model you pick from those installed (llama3.2, llama3.1, mistral, phi3, qwen2.5, qwen2.5-coder, deepseek-r1 — `<think>` blocks are stripped — plus any custom model). |
| 🧠 **Unlimited-OCR (Scanned PDFs, Local GPU)** | Pages rasterised at 200 DPI and parsed 8 pages per call by `baidu/Unlimited-OCR`; returns Markdown + tables, block types and bounding boxes. Non-PDF documents fall back to MarkItDown. |

### Image conversion method
| Mode | Behaviour |
|---|---|
| 🔤 **OCR (Free, No API Key)** | Tesseract. |
| 🤖 **Claude AI** | Claude describes the image and extracts text/tables. |
| 🦙 **Llama Vision (Local)** | An installed Ollama vision model (llava, llama3.2-vision, llava:13b, moondream, qwen2.5vl …). |
| 🧠 **Unlimited-OCR (Local GPU)** | Resolution modes: **Gundam** (tiled — dense/large pages) or **Base** (single 1024 px pass — faster). |

### Audio / video transcription
faster-whisper sizes `tiny · base · small · medium (default) · large-v3 · large-v3-turbo · distil-large-v3`, optional language code (blank = auto-detect). Runs on CUDA float16 when available, otherwise CPU int8.

## 8. Outputs

| Format | Renderer | Notes |
|---|---|---|
| `.md` | — | The Markdown as shown/edited |
| `.txt` | — | Markdown syntax stripped |
| `.json` | — | `{markdown, source, kind, sidecar}` — tables, cues, segments, page/bbox data |
| `.html` | `markdown` | Standalone styled HTML |
| `.docx` | `python-docx` | Headings, lists, tables, code blocks |
| `.pdf` | `fpdf2` | Unicode font auto-detected (Segoe UI / DejaVu / Arial fallback) |

Unlimited-OCR PDF runs also write `output/<file>.md` and `output/<file>.json` (page count, block types, tables as arrays, raw bounding boxes) next to the app.

## 9. Project structure

```
markitdown-app/
├─ Run MarkItDown.bat      # one-click launcher: venv, deps, GPU extras, port pick, browser open
├─ markitdown_app.py       # Streamlit UI: sidebar, uploads, URL flow, progress, preview, editor, downloads, history
├─ file_handlers.py        # per-type converters (subtitles, pptx, xlsx/csv, eml, zip, images, URLs, YouTube, AI enhance)
├─ unlimited_ocr.py        # Unlimited-OCR wrapper: staged warm-up generator, image/PDF OCR, sidecar + output/ writer
├─ whisper_handler.py      # faster-whisper: audio extraction (PyAV), CUDA/CPU selection, segment progress, Markdown
├─ output_formatter.py     # .md/.txt/.json/.html/.docx/.pdf renderers + zip bundle
├─ ollama_models.py        # discovers models installed in the local Ollama server (GET /api/tags)
├─ requirements.txt        # base dependencies
├─ requirements-ocr.txt    # optional GPU extras (torch cu128, transformers, faster-whisper, …)
└─ .streamlit/config.toml  # headless=true, dark theme (port is deliberately NOT here)
```

Each capability lives in its own Streamlit-free module so it can be tested directly, e.g.
`venv\Scripts\python -c "import file_handlers as fh; print(fh.convert_url('https://youtu.be/dQw4w9WgXcQ').markdown[:300])"`.

## 10. Configuration

| Setting | Where | Default |
|---|---|---|
| Port | picked by the launcher; override with env `STREAMLIT_SERVER_PORT` | random free port |
| Headless / theme | `.streamlit/config.toml` | `headless = true`, `base = "dark"` |
| Ollama server | env `OLLAMA_HOST` | `http://localhost:11434` |
| Unlimited-OCR model | `unlimited_ocr.MODEL_ID` | `baidu/Unlimited-OCR` |
| PDF rasterisation | `unlimited_ocr.PDF_DPI`, `PAGES_PER_CALL` | 200 DPI, 8 pages/call |
| History length | `markitdown_app.MAX_HISTORY` | 10 |
| Dependency re-install trigger | `DEPS_VERSION` in the launcher | bump when `requirements.txt` changes |

API keys are typed into the sidebar (`type="password"`) and live only in the session — nothing is written to disk.

## 11. Manual setup (no launcher / macOS / Linux)

```bash
python3.12 -m venv venv
# Windows: venv\Scripts\activate      macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
# optional GPU extras (NVIDIA only):
pip install -r requirements-ocr.txt
streamlit run markitdown_app.py --server.port 8502
```

Unlimited-OCR is CUDA-only (hard-coded upstream); on machines without an NVIDIA GPU the app simply hides that mode and everything else works.

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `[ERROR] Python is not installed` | Install Python 3.12 and tick *Add python.exe to PATH*. |
| Browser doesn't open | Open the URL printed in the launcher; see `%TEMP%\markitdown_launch.log`. |
| "Port already in use" | The launcher picks a free port each time; if you forced one via `STREAMLIT_SERVER_PORT`, choose another. |
| YouTube: *No captions available* | The video truly has no captions in en/hi/mr — use the inline uploader for Whisper. If **every** video fails, make sure `youtube-transcript-api >= 1.2.4` (older versions break on YouTube's current format). |
| Unlimited-OCR "not available" | Run `venv\Scripts\python -m pip install -r requirements-ocr.txt`; needs an NVIDIA GPU with ≥ 8 GB VRAM and an up-to-date driver. |
| Whisper slow | It's on CPU — install the GPU extras, or pick `small`/`base`. |
| Tesseract mode fails | Install Tesseract to the default path `C:\Program Files\Tesseract-OCR`. |
| Llama modes show no models | Start Ollama and `ollama pull llama3.2` (text) / `ollama pull llama3.2-vision` (images). |
| Folder lives in OneDrive and files show 0 bytes | OneDrive "files on demand" dehydrated them — right-click → *Always keep on this device*. |

## 13. Credits

- [microsoft/markitdown](https://github.com/microsoft/markitdown) — core conversion
- [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) — document parsing model
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — transcription
- [jdepoix/youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) — YouTube captions
- Streamlit, Playwright, PyMuPDF, python-pptx, pandas, python-docx, fpdf2, Ollama, Anthropic

UI and integration by **Kshitij**. Built with the help of Claude Code.

## License

[MIT](LICENSE) — free to use, modify and redistribute with attribution. Third-party models and libraries keep their own licenses.
