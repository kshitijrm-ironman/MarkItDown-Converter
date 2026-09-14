import streamlit as st

st.set_page_config(page_title="MarkItDown App - UI by Kshitij", page_icon="📝", layout="wide")

import base64
import io
import json
import os
import tempfile
import uuid
from datetime import datetime

import streamlit.components.v1 as components

import file_handlers as fh
import output_formatter as fmt
import unlimited_ocr
import ollama_models
import whisper_handler

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DOC_MODE_DEFAULT = fh.DOC_MODE_DEFAULT
DOC_MODE_CLAUDE = fh.DOC_MODE_CLAUDE
DOC_MODE_OLLAMA = fh.DOC_MODE_OLLAMA
UOCR_DOC_LABEL = "🧠 Unlimited-OCR (Scanned PDFs, Local GPU)"

IMAGE_MODE_OCR = "🔤 OCR (Free, No API Key)"
IMAGE_MODE_CLAUDE = "🤖 Claude AI (Needs API Key)"
IMAGE_MODE_OLLAMA = "🦙 Llama Vision (Local, Free)"
UOCR_IMAGE_LABEL = "🧠 Unlimited-OCR (Local GPU)"

# "What is this model best at?" notes shown under each model picker.
OLLAMA_TEXT_MODEL_NOTES = {
    "llama3.2": "Best at: quick, general-purpose structuring and summaries of everyday documents. "
                "3B params, ~2 GB — runs comfortably on modest CPUs/GPUs.",
    "llama3.1": "Best at: long reports and nuanced summaries — 128k context keeps large documents coherent. "
                "8B params, ~4.7 GB; slower than llama3.2 but noticeably more capable.",
    "mistral":  "Best at: concise, well-organised output for technical and business text (specs, contracts, emails). "
                "7B params, ~4.1 GB.",
    "phi3":     "Best at: very low-VRAM machines and short documents, lists and Q&A-style content. "
                "3.8B params, ~2.2 GB; weakest on very long inputs.",
    "qwen2.5":  "Best at: faithful, accurate restructuring of long or complex documents (reports, manuals, "
                "multilingual text) — the most capable general text model in the 14B class. ~9 GB; needs ~10 GB VRAM "
                "for full speed, otherwise partly runs on CPU (slow).",
    "qwen2.5-coder": "Best at: documents full of code, configs, logs and API docs — preserves code blocks, "
                     "indentation and symbols exactly. Weaker on plain prose than qwen2.5.",
    "deepseek-r1": "Best at: reasoning-heavy clean-ups — reconciling messy tables, inferring headings/structure "
                   "from unstructured text. Thinks step-by-step, so it is the slowest option (its hidden "
                   "<think> reasoning is stripped automatically).",
    "hf.co/invincibleambuj/Ambuj-Tripathi-Indian-Legal-Llama-GGUF": "Best at: Indian legal documents — contracts, judgments, statutes "
                   "(fine-tuned on Indian law). Only 1.2B params, so keep it to short documents.",
}
OLLAMA_VISION_MODEL_NOTES = {
    "llava":           "Best at: fast, all-round image description and reading clear text in screenshots/photos. "
                       "7B, ~4.7 GB — the quickest choice.",
    "llama3.2-vision": "Best at: charts, diagrams, tables and document layout — strongest visual reasoning of the four. "
                       "11B, ~7.9 GB; needs ~8 GB VRAM.",
    "llava:13b":       "Best at: more accurate descriptions and small-text reading than llava 7B, at roughly half the speed. "
                       "13B, ~8 GB.",
    "moondream":       "Best at: lightweight captions on weak hardware — simple photos, not dense text. "
                       "1.8B, ~1.7 GB; fastest, least detailed.",
    "qwen2.5vl":       "Best at: reading dense document text, tables and charts straight from images — the "
                       "strongest OCR-style vision model here, with a 128k context for multi-image jobs. "
                       "7B, ~6 GB.",
}

MAX_HISTORY = 10
MAX_PREVIEW_BYTES = 40 * 1024 * 1024
PREVIEW_CONVERT_EXTS = {".tif", ".tiff", ".heic", ".heif"}
AUDIO_MIME = {".mp3": "audio/mp3", ".wav": "audio/wav", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".flac": "audio/flac"}

OLLAMA_VISION_PROMPT = """You are a document analysis assistant.
                                            Analyze this image carefully and convert all content to clean Markdown format.
                                            Extract all text, describe diagrams, tables, charts.
                                            Format headings, lists, and tables properly in Markdown."""


# --------------------------------------------------------------------------- #
# Cached model loaders
# --------------------------------------------------------------------------- #
class StageProgress:
    """
    st.progress bar + st.empty() status line, updated together per stage.
    Percentages are 0-100. `done()` removes both widgets.
    """

    def __init__(self, icon: str = ""):
        self.icon = icon
        self.bar = st.empty()    # placeholders reserve the slot; nothing is drawn
        self.text = st.empty()   # until the first update()
        self.used = False

    def update(self, pct, msg: str) -> None:
        self.used = True
        pct = max(0, min(100, int(round(pct))))
        self.bar.progress(pct)
        self.text.markdown(f"{self.icon} **{pct}%** — {msg}")

    def done(self) -> None:
        self.bar.empty()
        self.text.empty()


def get_unlimited_ocr_model(prog: StageProgress | None = None):
    """
    Return the process-wide cached (model, tokenizer). If the startup warm-up
    did not run (or failed), this loads it now, reporting stages on `prog`.
    """
    return unlimited_ocr.load_model(progress=prog.update if prog else None)


def warm_unlimited_ocr_at_startup() -> None:
    """
    Warm the Unlimited-OCR model once per browser session, with a progress
    bar, only when the GPU backend is available. Tracked in session_state so
    reruns and later sessions (model already resident) don't show it again.
    """
    if ss.uocr_warmed:
        return
    if unlimited_ocr.is_loaded():          # another session already warmed it
        ss.uocr_warmed = True
        return
    available, detail = unlimited_ocr.availability()
    if not available:
        ss.uocr_warmed = True              # nothing to warm; don't re-check every rerun
        return

    status = st.status("🧠 Warming up Unlimited-OCR on the GPU...", expanded=True)

    def _set_status(**kw) -> None:
        # Outside a Streamlit script run (bare import / unit tests) st.status()
        # degrades to a plain container with no .update(); ignore that case.
        try:
            status.update(**kw)
        except Exception:
            pass

    with status:
        prog = StageProgress("🧠")
        try:
            for pct, msg, _payload in unlimited_ocr.warm_unlimited_ocr():
                prog.update(pct, msg)
            _set_status(label=f"🧠 {unlimited_ocr.WARMUP_STAGES[-1][1]} — {detail}", state="complete", expanded=False)
        except Exception as exc:
            ss.uocr_warm_error = str(exc)
            _set_status(label="⚠️ Unlimited-OCR warm-up failed", state="error", expanded=True)
            st.warning(f"Unlimited-OCR warm-up failed: {exc}\n\nThe model will be loaded again when you first pick an Unlimited-OCR mode.")
        finally:
            prog.done()
    ss.uocr_warmed = True


@st.cache_resource(show_spinner=False)
def get_whisper_model(size: str):
    """Load a faster-whisper model once per size (first call downloads the weights)."""
    return whisper_handler.load_model(size)


def render_unlimited_ocr_status():
    """Show GPU / installation status in the sidebar. Returns (available, detail)."""
    available, detail = unlimited_ocr.availability()
    if available:
        st.success(f"🎮 GPU ready: {detail}")
    else:
        st.error(f"⚠️ Unlimited-OCR unavailable: {detail}")
    st.caption("First use downloads the ~6.7 GB model from Hugging Face (cached afterwards).")
    return available, detail


def render_model_notes(selected: str, notes: dict[str, str], title: str = "What is each model best at?") -> None:
    """Caption for the selected model + an expander listing every model's strengths."""
    note = notes.get(selected)
    if note:
        st.caption(f"💡 **{selected}** — {note}")
    with st.expander(f"ℹ️ {title}", expanded=False):
        for name, text in notes.items():
            marker = "▶ " if name == selected else ""
            st.markdown(f"- {marker}**`{name}`** — {text}")


@st.cache_data(ttl=60, show_spinner=False)
def _local_ollama_models() -> tuple[list[ollama_models.LocalModel], str]:
    return ollama_models.list_local_models()


def ollama_model_picker(label: str, key: str, vision: bool, curated: dict[str, str], preferred: str) -> str:
    """
    Selectbox over the Ollama models installed on this machine (text or
    vision), each with a 'best at' note. Falls back to the curated list when
    the Ollama server is not reachable or has no model of that kind.
    """
    models, err = _local_ollama_models()
    text_models, vision_models = ollama_models.split_by_kind(models)
    installed = vision_models if vision else text_models
    kind = "vision" if vision else "text"

    if installed:
        by_name = {m.name: m for m in installed}
        names = list(by_name)
        # default: the preferred model if pulled (any tag), else the first one
        default = next((n for n in names if n == preferred or n.split(":")[0] == preferred), names[0])
        selected = st.selectbox(
            label, names, index=names.index(default), key=key,
            format_func=lambda n: by_name[n].label(),
            help=f"{len(installed)} local {kind} model(s) found in Ollama. Only models installed on this PC are listed.",
        )
        notes = {m.name: ollama_models.note_for(m, curated) for m in installed}
        st.caption(f"✅ {len(installed)} local {kind} model(s) found in Ollama")
        render_model_notes(selected, notes, title="What is each installed model best at?")
        not_pulled = {k: v for k, v in curated.items() if not any(m.base == k or m.name == k for m in models)}
        if not_pulled:
            with st.expander("⬇️ Other models you could pull", expanded=False):
                for name, text in not_pulled.items():
                    st.markdown(f"- **`{name}`** — {text}  \n  `ollama pull {name}`")
    else:
        names = list(curated)
        selected = st.selectbox(label, names, index=names.index(preferred) if preferred in names else 0, key=key,
                                help="Suggested models — pull one with `ollama pull <name>` first.")
        if err:
            st.warning(f"⚠️ {err}")
        else:
            st.warning(f"⚠️ No local {kind} model found in Ollama. Pull one first, e.g. `ollama pull {preferred}`.")
        render_model_notes(selected, curated)

    if st.button("🔄 Refresh model list", key=f"{key}_refresh", help="Re-read the installed models from Ollama"):
        _local_ollama_models.clear()
        st.rerun()
    return selected


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
ss = st.session_state
ss.setdefault("results", [])        # results of the latest conversion batch
ss.setdefault("active", 0)          # index of the result being viewed
ss.setdefault("history", [])        # last MAX_HISTORY conversions (newest first)
ss.setdefault("render_cache", {})   # (id, fmt, hash) -> bytes
ss.setdefault("uocr_warmed", False)  # Unlimited-OCR warm-up shown/finished this session
ss.setdefault("uocr_warm_error", "")


def human_size(n) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def make_result(conv: fh.ConversionResult, original: bytes | None, mode: str, saved_paths=None) -> dict:
    return {
        "id": uuid.uuid4().hex[:10],
        "name": conv.source_name,
        "kind": conv.kind,
        "markdown": conv.markdown,
        "original_markdown": conv.markdown,
        "sidecar": conv.sidecar,
        "notes": list(conv.notes),
        "bytes": original if (original is not None and len(original) <= MAX_PREVIEW_BYTES) else None,
        "size": len(original) if original is not None else None,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "saved_paths": list(saved_paths or []),
    }


def push_history(result: dict) -> None:
    entry = {k: result[k] for k in ("id", "name", "kind", "markdown", "sidecar", "notes", "time", "mode")}
    ss.history.insert(0, entry)
    del ss.history[MAX_HISTORY:]


def render_cached(result: dict, fmt_key: str) -> bytes:
    key = (result["id"], fmt_key, hash(result["markdown"]))
    cache = ss.render_cache
    if key not in cache:
        if len(cache) > 60:
            cache.clear()
        cache[key] = fmt.render(
            fmt_key,
            result["markdown"],
            source_name=result["name"],
            sidecar=result.get("sidecar"),
            extra={"kind": result["kind"], "conversion_mode": result.get("mode"), "converted_at": result["time"]},
        )
    return cache[key]


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def copy_button(text: str, key: str, label: str = "📋 Copy to clipboard") -> None:
    payload = json.dumps(text).replace("</", "<\\/")
    components.html(
        f"""
        <button id="copy_{key}" style="padding:6px 14px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer;font-size:14px">{label}</button>
        <script>
        (function() {{
          const btn = document.getElementById("copy_{key}");
          const text = {payload};
          btn.onclick = async () => {{
            try {{ await navigator.clipboard.writeText(text); }}
            catch (e) {{
              const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta);
              ta.select(); document.execCommand("copy"); ta.remove();
            }}
            btn.innerText = "✅ Copied!"; setTimeout(() => btn.innerText = {json.dumps(label)}, 1500);
          }};
        }})();
        </script>
        """,
        height=44,
    )


PDF_PREVIEW_DPI = 150


def pdf_page_cached(result: dict, page_index: int) -> bytes:
    """PNG of one PDF page, rasterised once per (result, page) and kept in session state."""
    key = (result["id"], "pdf_page", page_index)
    cache = ss.render_cache
    if key not in cache:
        cache[key] = fh.pdf_page_png(result["bytes"], page_index, dpi=PDF_PREVIEW_DPI)
    return cache[key]


def render_pdf_preview(result: dict) -> None:
    """Show a PDF as rasterised page images (PyMuPDF) with a page selector.

    Chrome refuses to load PDF iframes served from data:/blob: URLs on
    localhost, so rendering pages to PNG and using st.image bypasses that.
    """
    count_key = (result["id"], "pdf_pages")
    if count_key not in ss.render_cache:
        ss.render_cache[count_key] = fh.pdf_page_count(result["bytes"])
    total = ss.render_cache[count_key]
    if total == 0:
        st.info("This PDF has no pages.")
        return

    page = 1
    if total > 1:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total,
            value=1,
            step=1,
            key=f"pdf_page_{result['id']}",
            help=f"{total} pages — type a number or use the arrows.",
        )
    st.image(pdf_page_cached(result, int(page) - 1), use_container_width=True)
    st.caption(f"Page {int(page)} of {total} · rendered at {PDF_PREVIEW_DPI} DPI")


def render_original(result: dict) -> None:
    data = result.get("bytes")
    name = result["name"]
    ext = fh.ext_of(name)
    kind = result["kind"]

    if kind == "url":
        url = (result.get("sidecar") or {}).get("url", "")
        st.info(f"🌐 Source URL: {url}")
        if url:
            st.link_button("Open original page", url)
        return
    if data is None:
        why = " (files over 40 MB aren't kept in memory)" if result.get("size") else " (history items don't retain the original file)"
        st.info(f"Original preview not available{why}.")
        return
    try:
        if kind == "image":
            png = fh.image_preview_png(data, name) if ext in PREVIEW_CONVERT_EXTS else data
            st.image(png, use_container_width=True)
            if ext in fh.MULTIPAGE_IMAGE_EXTS:
                st.caption("Showing the first page of the TIFF.")
        elif ext == ".pdf":
            render_pdf_preview(result)
        elif kind == "audio":
            st.audio(data, format=AUDIO_MIME.get(ext, "audio/mpeg"))
        elif kind == "video":
            st.video(data)
        elif kind == "spreadsheet":
            import pandas as pd

            if ext == ".csv":
                sheets = {"CSV": pd.read_csv(io.BytesIO(data), sep=None, engine="python", dtype=str, keep_default_na=False)}
            else:
                sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str)
            for sheet_name, df in sheets.items():
                st.caption(f"Sheet: {sheet_name} — first {min(len(df), 200)} of {len(df)} rows")
                st.dataframe(df.head(200), use_container_width=True)
        elif kind == "archive":
            files = (result.get("sidecar") or {}).get("files", [])
            st.caption(f"Archive members ({len(files)}):")
            st.dataframe([{"File": f["name"], "Size": human_size(f["size_bytes"]), "Status": f["status"]} for f in files], use_container_width=True, hide_index=True)
        elif ext in fh.TEXT_PREVIEW_EXTS:
            text = data.decode("utf-8", errors="replace")
            st.code(text[:20000] + ("\n…(truncated)" if len(text) > 20000 else ""), language=None)
        else:
            st.info(f"No inline preview for {ext or 'this'} files ({human_size(len(data))}).")
    except Exception as exc:
        st.warning(f"Could not render a preview: {exc}")


# --------------------------------------------------------------------------- #
# Conversion dispatch
# --------------------------------------------------------------------------- #
def _ocr_fallback(pages: list[str], name: str) -> fh.ConversionResult:
    texts = fh.ocr_pages_tesseract(pages)
    return fh.ConversionResult(fh.join_pages(texts, name, "Extracted Text from Image"), name, "image")


def convert_image(path: str, name: str, s: dict, prog: StageProgress) -> fh.ConversionResult:
    mode = s["image_mode"]
    with tempfile.TemporaryDirectory(prefix="mdimg_") as work:
        pages = fh.image_pages(path, work)
        if len(pages) > 1:
            st.info(f"🖼️ Multi-page image: {len(pages)} pages detected.")

        # ---- Claude AI Mode ----
        if mode == IMAGE_MODE_CLAUDE:
            if not s["anthropic_api_key"]:
                raise RuntimeError("Please enter your Anthropic API key in the sidebar first.")
            try:
                from openai import OpenAI

                client = OpenAI(api_key=s["anthropic_api_key"], base_url="https://api.anthropic.com/v1/")
                with st.spinner("🤖 Claude AI is analysing the image..."):
                    texts = [fh.convert_with_markitdown(p, llm_client=client, llm_model="claude-sonnet-4-6") for p in pages]
                md = texts[0] if len(texts) == 1 else fh.join_pages(texts, name, f"Image Analysis: {name}", "Described by Claude AI")
                return fh.ConversionResult(md, name, "image")
            except Exception as ai_error:
                st.warning(f"⚠️ Claude AI failed: {ai_error}. Falling back to OCR...")
                return _ocr_fallback(pages, name)

        # ---- Llama Vision Mode ----
        if mode == IMAGE_MODE_OLLAMA:
            try:
                import ollama

                outputs = []
                with st.spinner(f"🦙 {s['ollama_model']} is analysing the image..."):
                    for p in pages:
                        with open(p, "rb") as img_file:
                            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                        response = ollama.chat(
                            model=s["ollama_model"],
                            messages=[{"role": "user", "content": OLLAMA_VISION_PROMPT, "images": [img_base64]}],
                        )
                        outputs.append(response["message"]["content"])
                md = f"# Image Analysis: {name}\n\n> Analyzed by {s['ollama_model']} (Local Ollama)\n\n"
                if len(outputs) == 1:
                    md += outputs[0]
                else:
                    md += "\n\n---\n\n".join(f"## Page {i}\n\n{t}" for i, t in enumerate(outputs, 1))
                return fh.ConversionResult(md, name, "image")
            except Exception as llama_error:
                st.warning(f"⚠️ Llama Vision failed: {llama_error}. Make sure Ollama is running! Falling back to OCR...")
                return _ocr_fallback(pages, name)

        # ---- Unlimited-OCR Mode ----
        if mode == UOCR_IMAGE_LABEL:
            if not s["uocr_available"]:
                st.error(f"❌ Unlimited-OCR is not available: {s['uocr_detail']} Using Tesseract OCR instead.")
                return _ocr_fallback(pages, name)
            try:
                model, tokenizer = get_unlimited_ocr_model(prog)   # instant if warmed at startup
                texts = []
                n = len(pages)
                for i, p in enumerate(pages, 1):
                    # Single image: stages land at 20/60/85%. Multi-page image:
                    # each page's stages are scaled into its share of 0-85%.
                    if n == 1:
                        cb = prog.update
                    else:
                        cb = lambda pct, msg, i=i: prog.update((i - 1 + pct / 100) / n * 85, f"Page {i} of {n} — {msg}")
                    texts.append(unlimited_ocr.ocr_image(model, tokenizer, p, mode=s["uocr_mode"], progress=cb))
                md = fh.join_pages(texts, name, f"Parsed Image: {name}", f"Parsed by Unlimited-OCR ({s['uocr_mode']} mode, local GPU)")
                return fh.ConversionResult(md, name, "image")
            except Exception as uocr_error:
                prog.done()
                st.warning(f"⚠️ Unlimited-OCR failed: {uocr_error}. Falling back to OCR...")
                return _ocr_fallback(pages, name)

        # ---- Default: Tesseract OCR ----
        with st.spinner("🔤 Tesseract OCR is extracting text..."):
            return _ocr_fallback(pages, name)


def convert_media(path: str, name: str, kind: str, s: dict, prog: StageProgress) -> fh.ConversionResult:
    ok, detail = whisper_handler.availability()
    if not ok:
        raise RuntimeError(detail)
    size = s["whisper_size"]
    ts = whisper_handler.format_timestamp

    prog.update(10, "Extracting audio track...")
    audio = whisper_handler.extract_audio(path)

    prog.update(20, f"Loading Whisper model ('{size}', first use downloads it)...")
    model, device = get_whisper_model(size)

    # faster-whisper's transcribe() is a lazy generator: each yielded segment
    # advances the bar through the 20% -> 95% band by audio time covered.
    def _progress(done, total, n_segments):
        prog.update(20 + done / total * 75, f"Transcribing {ts(done)} / {ts(total)} on {device.upper()} — {n_segments} segment(s)")

    result = whisper_handler.transcribe(model, audio, language=s["whisper_language"] or None, progress=_progress)

    prog.update(95, "Formatting transcript...")
    md = whisper_handler.to_markdown(result, name, kind, size, device)
    sidecar = {"engine": "faster-whisper", "model": size, "device": device, **result}
    return fh.ConversionResult(md, name, kind, sidecar=sidecar, notes=[f"Transcribed on {device.upper()} with faster-whisper '{size}'."])


def convert_document(path: str, name: str, s: dict, prog: StageProgress) -> tuple[fh.ConversionResult, list[str], bool]:
    """Returns (result, saved_paths, parsed_by_unlimited_ocr)."""
    ext = fh.ext_of(name)
    if s["doc_mode"] == UOCR_DOC_LABEL:
        if ext != ".pdf":
            st.info("ℹ️ Unlimited-OCR document mode only handles PDFs — using MarkItDown for this file.")
        elif not s["doc_uocr_available"]:
            st.error(f"❌ Unlimited-OCR is not available: {s['doc_uocr_detail']} Using MarkItDown instead.")
        else:
            try:
                model, tokenizer = get_unlimited_ocr_model(prog)   # instant if warmed at startup
                # ocr_pdf_detailed reports: 10% rasterizing, 10-85% per page, 90% assembling
                res = unlimited_ocr.ocr_pdf_detailed(model, tokenizer, path, progress=prog.update)
                final_md = f"# {os.path.splitext(name)[0]}\n\n> Parsed by Unlimited-OCR (local GPU)\n\n{res['markdown']}"
                sidecar = unlimited_ocr.build_sidecar(res, name)
                md_path, json_path = unlimited_ocr.save_outputs(final_md, sidecar, name)
                note = (
                    f"🧠 Parsed by Unlimited-OCR — {sidecar['page_count']} page(s), {sidecar['table_count']} table(s), "
                    f"blocks: {sidecar['block_type_counts']}. Saved to output/: {os.path.basename(md_path)}, {os.path.basename(json_path)}"
                )
                return fh.ConversionResult(final_md, name, "document", sidecar=sidecar, notes=[note]), [md_path, json_path], True
            except Exception as uocr_error:
                prog.done()
                st.warning(f"⚠️ Unlimited-OCR failed: {uocr_error}. Falling back to MarkItDown.")
    with st.spinner("⚡ MarkItDown is converting the document..."):
        raw = fh.convert_with_markitdown(path)
    return fh.ConversionResult(raw, name, "document"), [], False


AI_ENHANCE_KINDS = {"document", "subtitle", "presentation", "spreadsheet", "email", "audio", "video"}


def dispatch(path: str, name: str, s: dict, prog: StageProgress) -> tuple[fh.ConversionResult, list[str]]:
    """Convert one file on disk according to its type and the sidebar settings."""
    kind = fh.classify(name)
    saved: list[str] = []
    skip_ai = False

    if kind == "image":
        conv = convert_image(path, name, s, prog)
    elif kind in ("audio", "video"):
        conv = convert_media(path, name, kind, s, prog)
    elif kind == "subtitle":
        conv = fh.convert_subtitles(path, name)
    elif kind == "presentation":
        conv = fh.convert_pptx(path, name)
    elif kind == "spreadsheet":
        conv = fh.convert_spreadsheet(path, name)
    elif kind == "email":
        conv = fh.convert_email(path, name)
    elif kind == "archive":
        conv = fh.convert_archive(path, name, lambda p, n: dispatch(p, n, s, prog)[0])
    else:
        conv, saved, skip_ai = convert_document(path, name, s, prog)

    # ---- AI Enhancement if selected ----
    if kind in AI_ENHANCE_KINDS and not skip_ai and s["doc_mode"] in (DOC_MODE_CLAUDE, DOC_MODE_OLLAMA):
        engine = "Claude AI" if s["doc_mode"] == DOC_MODE_CLAUDE else f"{s['doc_ollama_model']} (Local)"
        if s["doc_mode"] == DOC_MODE_CLAUDE and not s["doc_anthropic_key"]:
            st.error("❌ Please enter your Anthropic API key in the sidebar. Showing raw conversion.")
        else:
            try:
                with st.spinner(f"🤖 {engine} is structuring and summarizing..."):
                    conv.markdown = fh.process_document_with_ai(
                        conv.markdown, name, s["doc_mode"],
                        anthropic_key=s["doc_anthropic_key"], ollama_model_name=s["doc_ollama_model"],
                    )
                conv.notes.append(f"✨ Enhanced by {engine} — structured and summarized.")
            except Exception as ai_error:
                st.warning(f"⚠️ {engine} failed: {ai_error}. Showing raw conversion.")
    return conv, saved


def convert_upload(name: str, data: bytes, s: dict) -> dict:
    suffix = fh.ext_of(name)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        prog = StageProgress("⏳")
        try:
            conv, saved = dispatch(tmp_path, name, s, prog)
            if prog.used:
                prog.update(100, "Generating output formats...")
            return make_result(conv, data, s["mode_label"], saved)
        finally:
            prog.done()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# --------------------------------------------------------------------------- #
# Sidebar — document settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("📄 Document Conversion Settings")
    doc_mode = st.radio(
        "Document conversion method:",
        [DOC_MODE_DEFAULT, DOC_MODE_CLAUDE, DOC_MODE_OLLAMA, UOCR_DOC_LABEL],
        help="Default uses MarkItDown. AI modes extract structure and summarize content. Unlimited-OCR parses PDF pages with a local vision model (best for scanned PDFs).",
    )
    doc_anthropic_key = None
    doc_ollama_model = "llama3.2"
    doc_uocr_available, doc_uocr_detail = False, ""
    if doc_mode == DOC_MODE_CLAUDE:
        doc_anthropic_key = st.text_input(
            "Anthropic API Key:", type="password", placeholder="sk-ant-...", key="doc_anthropic_key",
            help="Your key is never stored. Used only for this session.",
        )
        if doc_anthropic_key:
            st.success("✅ API Key received!")
    elif doc_mode == DOC_MODE_OLLAMA:
        doc_ollama_model = ollama_model_picker(
            "Select local text model:", key="doc_ollama_model_select", vision=False,
            curated=OLLAMA_TEXT_MODEL_NOTES, preferred="llama3.2",
        )
    elif doc_mode == UOCR_DOC_LABEL:
        doc_uocr_available, doc_uocr_detail = render_unlimited_ocr_status()
        st.caption("💡 **Unlimited-OCR** — Best at: scanned / image-only PDFs, dense tables and multi-column layouts. "
                   "Produces true Markdown (headings, tables) instead of a flat text dump. Needs an NVIDIA GPU.")

# defaults for conditional settings
image_mode = IMAGE_MODE_OCR
anthropic_api_key = None
ollama_model = "llava"
uocr_available, uocr_detail = False, ""
uocr_mode = "gundam"
whisper_size = whisper_handler.DEFAULT_MODEL
whisper_language = ""

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("📝 MarkItDown App - UI by Kshitij")
st.caption(
    "Convert documents, images, audio, video, subtitles, spreadsheets, presentations, e-mails, archives and URLs "
    "to Markdown — then export as .md / .txt / .json / .html / .docx / .pdf."
)

# One-time (per session) GPU warm-up with progress; no-op when unavailable or already resident.
warm_unlimited_ocr_at_startup()

input_mode = st.radio("Choose input method:", ["📁 Upload File", "🌐 Enter URL"], horizontal=True)

if input_mode == "📁 Upload File":
    uploaded_files = st.file_uploader(
        "Upload one or more files to convert",
        type=None,
        accept_multiple_files=True,
        help="Documents (PDF, DOCX, PPTX, XLSX, CSV, HTML…), images (JPG, PNG, WEBP, HEIC, TIFF…), audio (MP3, WAV, M4A, OGG, FLAC), "
             "video (MP4, MKV, AVI, MOV), subtitles (SRT, VTT), e-mail (EML) and ZIP archives.",
    )

    if uploaded_files:
        kinds = {fh.classify(f.name) for f in uploaded_files}
        st.dataframe(
            [{"File": f.name, "Type": fh.classify(f.name), "Size": human_size(f.size)} for f in uploaded_files],
            use_container_width=True, hide_index=True,
        )

        # ---- Sidebar: image settings (only when an image is in the batch) ----
        if "image" in kinds:
            with st.sidebar:
                st.divider()
                st.subheader("🖼️ Image Conversion Settings")
                image_mode = st.radio(
                    "Image conversion method:",
                    [IMAGE_MODE_OCR, IMAGE_MODE_CLAUDE, IMAGE_MODE_OLLAMA, UOCR_IMAGE_LABEL],
                    help="OCR extracts text. Claude AI describes images via API. Llama runs fully offline. Unlimited-OCR is Baidu's document-parsing model (needs an NVIDIA GPU). Multi-page TIFFs are processed page by page.",
                )
                if image_mode == IMAGE_MODE_CLAUDE:
                    anthropic_api_key = st.text_input(
                        "Enter Anthropic API Key:", type="password", placeholder="sk-ant-...", key="image_anthropic_key",
                        help="Your key is never stored. Used only for this session.",
                    )
                    if anthropic_api_key:
                        st.success("✅ API Key received!")
                elif image_mode == IMAGE_MODE_OLLAMA:
                    ollama_model = ollama_model_picker(
                        "Select local vision model:", key="image_ollama_model_select", vision=True,
                        curated=OLLAMA_VISION_MODEL_NOTES, preferred="llava",
                    )
                elif image_mode == UOCR_IMAGE_LABEL:
                    uocr_available, uocr_detail = render_unlimited_ocr_status()
                    uocr_mode = st.radio(
                        "Resolution mode:", list(unlimited_ocr.IMAGE_MODES),
                        format_func=lambda m: unlimited_ocr.IMAGE_MODE_LABELS[m],
                        help="Gundam tiles the image for dense/large pages. Base is a single 1024px pass and is faster.",
                    )
                    render_model_notes(uocr_mode, unlimited_ocr.IMAGE_MODE_NOTES, "What is each resolution mode best at?")

        # ---- Sidebar: audio/video settings ----
        if kinds & {"audio", "video"}:
            with st.sidebar:
                st.divider()
                st.subheader("🎙️ Audio / Video Transcription")
                w_ok, w_detail = whisper_handler.availability()
                (st.success if w_ok else st.error)(("🎮 " if w_ok else "⚠️ ") + w_detail)
                whisper_size = st.selectbox(
                    "faster-whisper model:", whisper_handler.MODEL_SIZES,
                    index=whisper_handler.MODEL_SIZES.index(whisper_handler.DEFAULT_MODEL),
                    format_func=whisper_handler.model_label,
                    help="Larger models are more accurate but slower and bigger to download (cached after first use).",
                )
                render_model_notes(whisper_size, whisper_handler.MODEL_BEST_AT, "What is each Whisper size best at?")
                whisper_language = st.text_input("Language code (blank = auto-detect):", placeholder="en, hi, de …", max_chars=5).strip().lower()
                st.caption("Video files: the audio track is decoded straight from the container (no ffmpeg needed).")

        if st.button("🔄 Convert to Markdown", type="primary", use_container_width=True):
            mode_bits = [doc_mode]
            if "image" in kinds:
                mode_bits.append(image_mode)
            if kinds & {"audio", "video"}:
                mode_bits.append(f"whisper:{whisper_size}")
            settings = dict(
                doc_mode=doc_mode, doc_anthropic_key=doc_anthropic_key, doc_ollama_model=doc_ollama_model,
                doc_uocr_available=doc_uocr_available, doc_uocr_detail=doc_uocr_detail,
                image_mode=image_mode, anthropic_api_key=anthropic_api_key, ollama_model=ollama_model,
                uocr_available=uocr_available, uocr_detail=uocr_detail, uocr_mode=uocr_mode,
                whisper_size=whisper_size, whisper_language=whisper_language,
                mode_label=" · ".join(mode_bits),
            )
            results: list[dict] = []
            total = len(uploaded_files)
            overall = st.progress(0, text=f"Converting {total} file(s)...")
            for idx, f in enumerate(uploaded_files):
                overall.progress(idx / total, text=f"Converting {f.name} ({idx + 1}/{total})...")
                with st.status(f"📄 {f.name}", expanded=True) as status:
                    try:
                        result = convert_upload(f.name, f.getvalue(), settings)
                        results.append(result)
                        push_history(result)
                        status.update(label=f"✅ {f.name} — {len(result['markdown']):,} characters", state="complete", expanded=False)
                    except Exception as exc:
                        st.error(f"❌ {f.name}: {exc}")
                        status.update(label=f"❌ {f.name} — failed", state="error", expanded=True)
            overall.empty()
            ss.results = results
            ss.active = 0
            ss.render_cache = {}
            if results:
                st.success(f"✅ Converted {len(results)} of {total} file(s). Scroll down for results.")
            else:
                st.error("No files were converted.")

elif input_mode == "🌐 Enter URL":
    url_input = st.text_input(
        "Enter a website URL:",
        placeholder="https://EXAMPLE.COM, https://en.wikipedia.org/wiki/Concrete or a YouTube link",
        help="Web pages, Wikipedia articles and YouTube videos (captions: English → Hindi → Marathi).",
    ).strip()
    if url_input:
        st.success(f"✅ URL Received: **{url_input}**")
        yt_id = fh.extract_youtube_id(url_input)
        if yt_id:
            st.caption(f"▶️ YouTube video detected (id `{yt_id}`) — captions will be fetched directly, English → Hindi → Marathi.")
        if st.button("🔄 Convert to Markdown", type="primary", use_container_width=True):
            ss.yt_no_captions = None
            with st.spinner("⏳ Fetching YouTube captions..." if yt_id else "🤖 MarkItDown Converting your file...Please Wait"):
                try:
                    conv = fh.convert_url(url_input, status=st.info)
                    result = make_result(conv, None, "YouTube captions" if yt_id else "URL")
                    push_history(result)
                    ss.results = [result]
                    ss.active = 0
                    ss.render_cache = {}
                    if yt_id:
                        sc = conv.sidecar or {}
                        st.success(f"✅ Transcript fetched — {sc.get('segment_count', 0)} caption cue(s), language: {sc.get('language', '?')}"
                                   f"{' (auto-generated)' if sc.get('auto_generated') else ''}.")
                    else:
                        st.success("✅ URL converted successfully!")
                except fh.YouTubeNoCaptionsError as exc:
                    # Remembered in session state so the inline uploader below survives the reruns
                    # that Streamlit triggers while the user picks a file.
                    ss.yt_no_captions = {"url": url_input, "detail": exc.detail}
                except fh.URLConversionError as exc:
                    st.error(str(exc))

    # ---- No-captions fallback: exact message + inline Whisper uploader ----
    yt_fail = ss.get("yt_no_captions")
    if yt_fail and yt_fail["url"] == url_input:
        st.error(fh.YOUTUBE_NO_CAPTIONS_MSG)
        if yt_fail.get("detail"):
            st.caption(f"Details: {yt_fail['detail'][:300]}")

        with st.sidebar:
            st.divider()
            st.subheader("🎙️ Audio / Video Transcription")
            w_ok, w_detail = whisper_handler.availability()
            (st.success if w_ok else st.error)(("🎮 " if w_ok else "⚠️ ") + w_detail)
            yt_whisper_size = st.selectbox(
                "faster-whisper model:", whisper_handler.MODEL_SIZES,
                index=whisper_handler.MODEL_SIZES.index(whisper_handler.DEFAULT_MODEL),
                format_func=whisper_handler.model_label, key="yt_whisper_size",
                help="Larger models are more accurate but slower and bigger to download (cached after first use).",
            )
            render_model_notes(yt_whisper_size, whisper_handler.MODEL_BEST_AT, "What is each Whisper size best at?")
            yt_whisper_language = st.text_input(
                "Language code (blank = auto-detect):", placeholder="en, hi, de …", max_chars=5, key="yt_whisper_lang",
            ).strip().lower()

        yt_file = st.file_uploader(
            "Upload the video (or its audio) here and it will be transcribed with faster-whisper:",
            type=[e.lstrip(".") for e in sorted(fh.VIDEO_EXTS | fh.AUDIO_EXTS)],
            accept_multiple_files=False, key="yt_fallback_upload",
        )
        if yt_file is not None:
            stamp = (yt_fail["url"], yt_file.name, yt_file.size, yt_whisper_size, yt_whisper_language)
            if ss.get("yt_fallback_done") != stamp:      # don't re-transcribe on every rerun
                settings = dict(
                    doc_mode=doc_mode, doc_anthropic_key=doc_anthropic_key, doc_ollama_model=doc_ollama_model,
                    doc_uocr_available=doc_uocr_available, doc_uocr_detail=doc_uocr_detail,
                    image_mode=None, anthropic_api_key=None, ollama_model=None,
                    uocr_available=False, uocr_detail="", uocr_mode=None,
                    whisper_size=yt_whisper_size, whisper_language=yt_whisper_language,
                    mode_label=f"{doc_mode} · whisper:{yt_whisper_size} (YouTube fallback)",
                )
                with st.status(f"🎬 {yt_file.name}", expanded=True) as status:
                    try:
                        result = convert_upload(yt_file.name, yt_file.getvalue(), settings)
                        push_history(result)
                        ss.results = [result]
                        ss.active = 0
                        ss.render_cache = {}
                        ss.yt_fallback_done = stamp
                        status.update(label=f"✅ {yt_file.name} — {len(result['markdown']):,} characters", state="complete", expanded=False)
                    except Exception as exc:
                        st.error(f"❌ {yt_file.name}: {exc}")
                        status.update(label=f"❌ {yt_file.name} — failed", state="error", expanded=True)


# --------------------------------------------------------------------------- #
# Results: side-by-side preview, editor, downloads
# --------------------------------------------------------------------------- #
def _apply_edit(result_id: str, widget_key: str) -> None:
    for r in ss.results:
        if r["id"] == result_id:
            r["markdown"] = ss[widget_key]
            break


results = ss.results
if results:
    st.divider()
    st.subheader("📋 Results")
    if len(results) > 1:
        ss.active = st.selectbox(
            "Select a converted file:", range(len(results)),
            format_func=lambda i: f"{i + 1}. {results[i]['name']}  ({results[i]['kind']})",
            index=min(ss.active, len(results) - 1),
        )
    r = results[min(ss.active, len(results) - 1)]
    for note in r["notes"]:
        st.info(note)
    st.caption(f"Converted {r['time']} · mode: {r['mode']} · {len(r['markdown']):,} characters")

    tab_preview, tab_edit, tab_download = st.tabs(["👀 Side-by-side Preview", "✏️ Edit", "⬇️ Download"])

    with tab_preview:
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("**Original**")
            with st.container(height=640):
                render_original(r)
        with right:
            st.markdown("**Markdown**")
            copy_button(r["markdown"], key=f"prev_{r['id']}")
            with st.container(height=590):
                st.markdown(r["markdown"])

    with tab_edit:
        editor_key = f"editor_{r['id']}"
        if editor_key not in ss:
            ss[editor_key] = r["markdown"]
        st.text_area(
            "Edit the Markdown — changes are used by every download format:",
            key=editor_key, height=520, on_change=_apply_edit, args=(r["id"], editor_key),
        )
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            copy_button(r["markdown"], key=f"edit_{r['id']}")
        with c2:
            if st.button("↩️ Reset edits", key=f"reset_{r['id']}", use_container_width=True):
                r["markdown"] = r["original_markdown"]
                ss[editor_key] = r["original_markdown"]
                st.rerun()
        with c3:
            if r["markdown"] != r["original_markdown"]:
                st.caption("✏️ Edited — downloads use your edited text.")

    with tab_download:
        st.markdown("**Choose output format(s):**")
        cols = st.columns(3)
        selected: list[str] = []
        for i, (key, (label, mime, ext)) in enumerate(fmt.FORMATS.items()):
            with cols[i % 3]:
                if st.checkbox(label, value=(key in fmt.DEFAULT_FORMATS), key=f"fmt_{key}"):
                    selected.append(key)
        if not selected:
            st.warning("Select at least one output format.")
        else:
            st.markdown(f"**{r['name']}**")
            for key in selected:
                label, mime, ext = fmt.FORMATS[key]
                try:
                    data = render_cached(r, key)
                except Exception as exc:
                    st.error(f"Could not render {label}: {exc}")
                    continue
                st.download_button(
                    f"⬇️ Download {label}", data=data, file_name=fmt.filename_for(r["name"], key),
                    mime=mime, use_container_width=True, key=f"dl_{r['id']}_{key}",
                )
            if len(selected) > 1 or len(results) > 1:
                items: list[tuple[str, bytes]] = []
                failures: list[str] = []
                for rr in results:
                    for key in selected:
                        try:
                            items.append((fmt.filename_for(rr["name"], key), render_cached(rr, key)))
                        except Exception as exc:
                            failures.append(f"{rr['name']} → {key}: {exc}")
                if items:
                    st.download_button(
                        f"📦 Download all — {len(results)} file(s) × {len(selected)} format(s) as ZIP",
                        data=fmt.zip_bundle(items),
                        file_name=f"markitdown_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip", use_container_width=True, key=f"dl_zip_{r['id']}",
                    )
                for msg in failures:
                    st.warning(msg)
            if key == "json" or "json" in selected:
                if r.get("sidecar"):
                    st.caption("The JSON export embeds this file's structured data (tables, pages, blocks/bboxes, cues or transcript segments).")
        if r["saved_paths"]:
            st.caption("Also saved on disk: " + ", ".join(f"`{p}`" for p in r["saved_paths"]))


# --------------------------------------------------------------------------- #
# Sidebar — history (rendered last so this run's conversions are included)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.divider()
    with st.expander(f"🕘 History (last {MAX_HISTORY})", expanded=False):
        if not ss.history:
            st.caption("No conversions yet.")
        for h in ss.history:
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{h['name']}**<br><small>{h['kind']} · {h['time']}</small>", unsafe_allow_html=True)
            if c2.button("Load", key=f"hist_{h['id']}"):
                restored = dict(h, bytes=None, size=None, saved_paths=[], original_markdown=h["markdown"])
                ss.results = [restored]
                ss.active = 0
                ss.render_cache = {}
                st.rerun()
        if ss.history and st.button("🗑️ Clear history", use_container_width=True):
            ss.history = []
            st.rerun()
