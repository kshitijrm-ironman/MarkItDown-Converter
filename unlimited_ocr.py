"""
Unlimited-OCR wrapper (https://github.com/baidu/Unlimited-OCR).

Optional, GPU-only backend for the MarkItDown app. This module never imports
torch/transformers at import time, so the app still starts when they are not
installed. Call `availability()` first; load the model with `warm_unlimited_ocr()` (a
generator that yields progress) or `load_model()` (blocking) — both cache
the result process-wide — and then use `ocr_image()` / `ocr_pdf()`.

Facts about the upstream remote code this wrapper relies on:
  * `model.infer(...)` only RETURNS the decoded text when `eval_mode=True`
    (otherwise it returns None and just prints / writes files).
  * `model.infer_multi(...)` always returns `(text, n_tokens)`; pages are
    separated by the literal token `<PAGE>`. It does not support crop mode.
  * The language model only ships "eager" and "flash_attention_2" attention
    kernels and uses non-MLA attention, so the only valid choice on Windows
    (no flash-attn wheel) is `attn_implementation="eager"`.
  * CUDA is hard-coded inside `infer`, so a CPU fallback is not possible.
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
import os
import re
import tempfile
import threading as _threading
import time as _time
from html.parser import HTMLParser

MODEL_ID = "baidu/Unlimited-OCR"

# Where `save_outputs()` writes the .md + .json pair: <app dir>/output
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Resolution presets from the upstream README.
IMAGE_MODES = {
    "gundam": dict(base_size=1024, image_size=640, crop_mode=True),
    "base": dict(base_size=1024, image_size=1024, crop_mode=False),
}
IMAGE_MODE_LABELS = {
    "gundam": "Gundam (tiled, best for dense/large pages)",
    "base": "Base (single 1024px view, faster)",
}
IMAGE_MODE_NOTES = {
    "gundam": "Best at: full scanned pages, small print, dense tables and multi-column layouts — "
              "the image is tiled at 640px plus a 1024px global view, so nothing is downscaled away. "
              "Slower and uses more VRAM.",
    "base":   "Best at: screenshots, slides, receipts and other single-block images with normal-sized text — "
              "one 1024px pass, roughly 2-3× faster than Gundam. Tiny text on large pages may be lost.",
}

# Generation settings from the upstream README.
MAX_LENGTH = 32768
NO_REPEAT_NGRAM = 35
NGRAM_WINDOW_IMAGE = 128
NGRAM_WINDOW_MULTI = 1024

PDF_DPI = 200
PAGES_PER_CALL = 8  # keep each infer_multi call well inside the 32k context

STOP_TOKEN = "<｜end▁of▁sentence｜>"

# <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>  -> captured label
_REF_DET_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>", re.S)
# bare <|det|>label [[...]]<|/det|>
_DET_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.S)
_BLANK_RUN_RE = re.compile(r"\n{3,}")


# --------------------------------------------------------------------------- #
# Availability / model loading
# --------------------------------------------------------------------------- #
def availability() -> tuple[bool, str]:
    """Return (ok, detail). `detail` is a GPU description or a reason it's unavailable."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "PyTorch is not installed. Run 'Run MarkItDown.bat' and accept the Unlimited-OCR setup, or: pip install -r requirements-ocr.txt"
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False, "transformers is not installed. Run: pip install -r requirements-ocr.txt"
    try:
        import fitz  # noqa: F401  (PyMuPDF, used for PDF rasterisation)
    except ImportError:
        return False, "PyMuPDF is not installed. Run: pip install -r requirements-ocr.txt"

    if not torch.cuda.is_available():
        return False, "No CUDA GPU detected. Unlimited-OCR requires an NVIDIA GPU (the model hard-codes .cuda())."

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    detail = f"{props.name} ({vram_gb:.1f} GB VRAM)"
    if vram_gb < 8:
        return False, f"{detail} — the bf16 weights alone are ~6.7 GB; at least 8 GB VRAM is needed."
    return True, detail


# Process-wide cache of the loaded (model, tokenizer). Streamlit re-runs the
# script per interaction but keeps modules imported, so this survives reruns
# and is shared by every browser session on the same server process.
_LOADED: tuple | None = None
_LOAD_LOCK = _threading.Lock()

# (percent, message) for each warm-up stage, in order.
WARMUP_STAGES = (
    (0, "Loading model config..."),
    (10, "Loading tokenizer..."),
    (25, "Loading model weights onto CPU..."),
    (60, "Transferring weights to GPU..."),
    (80, "Compiling CUDA kernels (first run only)..."),
    (95, "Running warm-up inference..."),
    (100, "Unlimited-OCR ready on GPU"),
)


def is_loaded() -> bool:
    return _LOADED is not None


def _warmup_image(out_dir: str) -> str:
    """A small synthetic 'document' so the warm-up pass exercises the real pipeline."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (512, 256), "white")
    draw = ImageDraw.Draw(im)
    draw.text((24, 24), "Unlimited-OCR warm-up", fill="black")
    draw.text((24, 64), "Hello world. 1234567890", fill="black")
    path = os.path.join(out_dir, "warmup.png")
    im.save(path, "PNG")
    return path


def _from_pretrained(loader, **kwargs):
    """
    Load from the local Hugging Face cache first; only contact the Hub when the
    files are not cached yet (the one-time ~6.7 GB download).

    Without `local_files_only=True`, transformers makes a HEAD request per file
    to check for updates even when everything is cached. With no internet those
    requests retry for minutes (measured: config + tokenizer alone took ~2.5 min
    before falling back to the cache), which is why the app looked stuck offline.
    """
    try:
        return loader(MODEL_ID, local_files_only=True, **kwargs)
    except OSError:
        # Not in the cache yet (first run) — download from the Hub.
        return loader(MODEL_ID, **kwargs)


def warm_unlimited_ocr():
    """
    Generator that loads Unlimited-OCR onto the GPU while yielding progress.

    Yields `(percent, message, payload)` for each stage in `WARMUP_STAGES`;
    `payload` is None until the final 100% step, where it is `(model, tokenizer)`.
    Drive it with `for pct, msg, payload in warm_unlimited_ocr(): ...`.

    Safe to call repeatedly: once loaded, it yields only the final step. If
    another thread is mid-load, this waits for it (yielding a waiting message
    first) instead of loading a second copy onto the GPU.
    """
    global _LOADED

    if _LOADED is not None:
        yield 100, WARMUP_STAGES[-1][1], _LOADED
        return

    if _LOAD_LOCK.locked():
        yield 0, "Waiting for another session's warm-up to finish...", None
    with _LOAD_LOCK:
        if _LOADED is not None:  # loaded by whoever held the lock
            yield 100, WARMUP_STAGES[-1][1], _LOADED
            return

        yield WARMUP_STAGES[0][0], WARMUP_STAGES[0][1], None
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        config = _from_pretrained(AutoConfig.from_pretrained, trust_remote_code=True)

        yield WARMUP_STAGES[1][0], WARMUP_STAGES[1][1], None
        tokenizer = _from_pretrained(AutoTokenizer.from_pretrained, trust_remote_code=True)

        yield WARMUP_STAGES[2][0], WARMUP_STAGES[2][1], None
        model = _from_pretrained(
            AutoModel.from_pretrained,
            config=config,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        yield WARMUP_STAGES[3][0], WARMUP_STAGES[3][1], None
        model = model.eval().cuda()
        torch.cuda.synchronize()

        yield WARMUP_STAGES[4][0], WARMUP_STAGES[4][1], None
        # First bf16 matmul on the device creates the cuBLAS handle / workspace
        # and triggers lazy CUDA module loading for the kernels the LM uses.
        with torch.no_grad():
            a = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
            (a @ a).sum().item()
        torch.cuda.synchronize()

        yield WARMUP_STAGES[5][0], WARMUP_STAGES[5][1], None
        # A real (tiny) inference: hits the vision encoder + generation path so
        # cuDNN autotuning and remaining kernel loads happen now, not on the
        # user's first document.
        with tempfile.TemporaryDirectory(prefix="uocr_warm_") as work:
            png = _warmup_image(work)
            with torch.no_grad():
                model.infer(
                    tokenizer,
                    prompt="<image>document parsing.",
                    image_file=png,
                    output_path=work,
                    # max_length counts the prompt too (the image alone is ~280
                    # tokens in base mode); generation stops at EOS long before this.
                    max_length=1024,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM,
                    ngram_window=NGRAM_WINDOW_IMAGE,
                    eval_mode=True,
                    save_results=False,
                    **IMAGE_MODES["base"],
                )
        torch.cuda.synchronize()

        _LOADED = (model, tokenizer)
        yield WARMUP_STAGES[6][0], WARMUP_STAGES[6][1], _LOADED


def load_model(progress=None):
    """
    Download (first time, ~6.7 GB) and load the model onto the GPU. After the
    first run it loads purely from the local cache, so it works offline.
    Returns (model, tokenizer). Cached process-wide after the first call.
    `progress(percent, message)` is called for each warm-up stage if given.
    """
    payload = _LOADED
    for pct, msg, payload in warm_unlimited_ocr():
        if progress:
            progress(pct, msg)
    return payload


# --------------------------------------------------------------------------- #
# Output post-processing
# --------------------------------------------------------------------------- #
def clean_output(raw: str) -> str:
    """Strip layout/bbox markers and stop tokens so only Markdown remains."""
    if not raw:
        return ""
    text = raw.replace(STOP_TOKEN, "")

    def _ref_sub(match: re.Match) -> str:
        label = match.group(1).strip().lower()
        return "*[image]*\n" if label == "image" else ""

    text = _REF_DET_RE.sub(_ref_sub, text)
    text = _DET_RE.sub("", text)
    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Structured extraction (blocks / bboxes / tables) for the JSON sidecar
# --------------------------------------------------------------------------- #
# Same marker pair as _REF_DET_RE but captures the det payload too.
_REF_DET_FULL_RE = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*<\|det\|>(?P<det>.*?)<\|/det\|>", re.S
)
# Markdown table separator row:  | --- | :---: | ---: |
_MD_TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


def _block_type(label: str) -> str:
    """Map the model's free-form layout label onto text / table / figure."""
    low = label.strip().lower()
    if "table" in low:
        return "table"
    if any(k in low for k in ("image", "figure", "picture", "chart", "photo", "graph")):
        return "figure"
    return "text"


def _parse_bboxes(det_raw: str):
    """'[[x1, y1, x2, y2], ...]' -> list of int lists, or None if unparseable."""
    try:
        parsed = ast.literal_eval(det_raw.strip())
        if isinstance(parsed, (list, tuple)):
            return [[int(round(float(v))) for v in box] for box in parsed]
    except Exception:
        pass
    return None


def parse_blocks(raw: str) -> list[dict]:
    """
    Layout blocks in reading order, taken from the RAW model output (i.e. before
    `clean_output` strips the <|ref|>/<|det|> markers).

    Each block: {index, type, label, bboxes, det_raw, content}
      * type     - normalised: "text" | "table" | "figure"
      * label    - the model's own label (e.g. "title", "text", "table", "image")
      * bboxes   - parsed [[x1, y1, x2, y2], ...] (model coords, 0-999 grid) or None
      * det_raw  - the exact string between <|det|> and <|/det|>, untouched
      * content  - the block's text as emitted by the model (markers removed)
    """
    raw = raw or ""
    matches = list(_REF_DET_FULL_RE.finditer(raw))
    blocks: list[dict] = []
    for i, m in enumerate(matches):
        label = m.group("label").strip()
        det_raw = m.group("det").strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        content = raw[m.end() : end].replace(STOP_TOKEN, "").strip()
        blocks.append(
            {
                "index": i,
                "type": _block_type(label),
                "label": label,
                "bboxes": _parse_bboxes(det_raw),
                "det_raw": det_raw,
                "content": content,
            }
        )
    return blocks


class _TableHTMLParser(HTMLParser):
    """Collect every <table> as a list of rows, each row a list of cell strings."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._rows = None
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._rows = []
        elif tag == "tr" and self._rows is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self._rows.append(self._row)
            self._row = None
        elif tag == "table" and self._rows is not None:
            self.tables.append(self._rows)
            self._rows = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _markdown_tables(text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines() + [""]:
        s = line.strip()
        if len(s) > 1 and s.startswith("|") and s.endswith("|"):
            if _MD_TABLE_SEP_RE.match(s):
                continue
            current.append([c.strip() for c in s[1:-1].split("|")])
        else:
            if len(current) >= 2:
                tables.append(current)
            current = []
    return tables


def extract_tables(text: str) -> list[dict]:
    """
    All tables in `text` as arrays: [{"format": "html"|"markdown", "rows": [[cell, ...], ...]}].
    Unlimited-OCR emits tables as HTML; Markdown pipe tables are handled too.
    """
    text = text or ""
    found: list[dict] = []
    if "<table" in text.lower():
        parser = _TableHTMLParser()
        parser.feed(text)
        parser.close()
        found.extend({"format": "html", "rows": rows} for rows in parser.tables if rows)
    found.extend({"format": "markdown", "rows": rows} for rows in _markdown_tables(text))
    return found


def build_sidecar(details: dict, source_name: str) -> dict:
    """
    Assemble the JSON sidecar from `ocr_pdf_detailed()` output:
    page count, block types, tables as arrays, and raw bbox data.
    """
    pages_out: list[dict] = []
    all_tables: list[dict] = []
    raw_bboxes: list[dict] = []
    type_counts = {"text": 0, "table": 0, "figure": 0}

    for page in details["pages"]:
        page_no = page["page"]
        for b in page["blocks"]:
            type_counts[b["type"]] = type_counts.get(b["type"], 0) + 1
            raw_bboxes.append(
                {
                    "page": page_no,
                    "block_index": b["index"],
                    "type": b["type"],
                    "label": b["label"],
                    "det_raw": b["det_raw"],
                    "bboxes": b["bboxes"],
                }
            )
        page_tables = [
            {"page": page_no, "index": i, "format": t["format"], "rows": t["rows"]}
            for i, t in enumerate(page["tables"])
        ]
        all_tables.extend(page_tables)
        pages_out.append(
            {
                "page": page_no,
                "block_types": [b["type"] for b in page["blocks"]],
                "blocks": page["blocks"],
                "tables": page_tables,
            }
        )

    return {
        "source_file": source_name,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "engine": "Unlimited-OCR",
        "model": MODEL_ID,
        "dpi": details.get("dpi", PDF_DPI),
        "bbox_coordinate_space": "model grid 0-999 (relative to the rendered page image)",
        "page_count": details["page_count"],
        "parsed_page_count": len(details["pages"]),
        "block_type_counts": type_counts,
        "table_count": len(all_tables),
        "generation": details.get("stats"),
        "tables": all_tables,
        "pages": pages_out,
        "raw_bboxes": raw_bboxes,
    }


def _safe_stem(source_name: str) -> str:
    stem = os.path.splitext(os.path.basename(source_name or ""))[0]
    # Only replace characters that are illegal in Windows/macOS/Linux filenames.
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem).strip(" .")
    return stem or "document"


def save_outputs(markdown: str, sidecar: dict, source_name: str, output_dir: str = OUTPUT_DIR) -> tuple[str, str]:
    """
    Write <output_dir>/<stem>.md and <stem>.json (overwriting any previous pair
    with the same stem). Returns (md_path, json_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = _safe_stem(source_name)
    md_path = os.path.join(output_dir, f"{stem}.md")
    json_path = os.path.join(output_dir, f"{stem}.json")
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(markdown)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, ensure_ascii=False, indent=2)
    return md_path, json_path


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def _to_rgb_png(image_path: str, out_dir: str) -> str:
    """Normalise any PIL-readable image (jfif/gif/tiff/webp/...) to an RGB PNG."""
    from PIL import Image

    out = os.path.join(out_dir, "input.png")
    with Image.open(image_path) as im:
        im.convert("RGB").save(out, "PNG")
    return out


def _report(progress, pct: int, msg: str) -> None:
    if progress:
        progress(pct, msg)


# Stage percentages for single-image OCR (see `ocr_image`).
IMAGE_STAGES = (
    (20, "Preprocessing image..."),
    (60, "Running OCR inference..."),
    (85, "Cleaning output markers..."),
)


def _gen_stats(tokens: int, seconds: float) -> dict:
    """Generation-speed summary: {generated_tokens, seconds, tokens_per_sec}."""
    return {
        "generated_tokens": int(tokens),
        "seconds": round(seconds, 1),
        "tokens_per_sec": round(tokens / seconds, 1) if seconds > 0 else 0.0,
    }


def format_stats(stats: dict) -> str:
    """One-line human-readable speed read-out for the UI."""
    return (f"{stats['generated_tokens']:,} tokens generated in {stats['seconds']:.1f} s "
            f"({stats['tokens_per_sec']:.1f} tok/s)")


def ocr_image_detailed(model, tokenizer, image_path: str, mode: str = "gundam", progress=None) -> dict:
    """
    Parse a single image to Markdown.
    Returns {"markdown", "stats"}; `stats` is a generation-speed summary
    (token count is measured by re-tokenising the output, since `infer()`
    only returns text).
    `progress(percent, message)` is called at each `IMAGE_STAGES` step if given.
    """
    if mode not in IMAGE_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {list(IMAGE_MODES)}")

    with tempfile.TemporaryDirectory(prefix="uocr_") as work:
        _report(progress, *IMAGE_STAGES[0])
        png = _to_rgb_png(image_path, work)
        _report(progress, *IMAGE_STAGES[1])
        t0 = _time.perf_counter()
        raw = model.infer(
            tokenizer,
            prompt="<image>document parsing.",
            image_file=png,
            output_path=work,
            max_length=MAX_LENGTH,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            ngram_window=NGRAM_WINDOW_IMAGE,
            eval_mode=True,       # required: only this path returns the text
            save_results=False,
            **IMAGE_MODES[mode],
        )
        seconds = _time.perf_counter() - t0
    _report(progress, *IMAGE_STAGES[2])
    raw = raw or ""
    try:
        n_tokens = len(tokenizer.encode(raw, add_special_tokens=False))
    except Exception:
        n_tokens = 0
    return {"markdown": clean_output(raw), "stats": _gen_stats(n_tokens, seconds)}


def ocr_image(model, tokenizer, image_path: str, mode: str = "gundam", progress=None) -> str:
    """Markdown-only convenience wrapper around `ocr_image_detailed()`."""
    return ocr_image_detailed(model, tokenizer, image_path, mode=mode, progress=progress)["markdown"]


def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = PDF_DPI) -> list[str]:
    """Rasterise every page of a PDF to PNG (port of the upstream README helper)."""
    import fitz  # PyMuPDF

    paths: list[str] = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            out = os.path.join(out_dir, f"page_{i + 1:04d}.png")
            page.get_pixmap(matrix=matrix).save(out)
            paths.append(out)
    return paths


def _page_record(page_no: int, raw: str) -> dict:
    markdown = clean_output(raw)
    return {
        "page": page_no,
        "markdown": markdown,
        "blocks": parse_blocks(raw),   # from RAW text, before marker stripping
        "tables": extract_tables(markdown),
    }


def ocr_pdf_detailed(
    model,
    tokenizer,
    pdf_path: str,
    dpi: int = PDF_DPI,
    pages_per_call: int = PAGES_PER_CALL,
    progress=None,
) -> dict:
    """
    Parse a (scanned) PDF with multi-page inference and keep the structure.

    Returns {"markdown", "page_count", "dpi", "pages": [{page, markdown, blocks, tables}]}.
    `page_count` is the PDF's real page count; `pages` holds one record per
    chunk the model returned (normally identical).

    `progress(percent, message)` is called if given:
      * 10%  "Rasterizing PDF pages..."
      * 10% + N/M*75%  "Processing page N of M..." — before each batch (N =
        first page of the batch) and after it (N = last page of the batch).
        Pages are sent to the model `pages_per_call` at a time, so progress
        advances in steps of that size.
      * 90%  "Assembling output..."
    """
    page_records: list[dict] = []
    total_tokens = 0
    gen_seconds = 0.0
    with tempfile.TemporaryDirectory(prefix="uocr_pdf_") as work:
        _report(progress, 10, "Rasterizing PDF pages...")
        pages = pdf_to_images(pdf_path, work, dpi=dpi)
        total = len(pages)

        def page_pct(n: int) -> int:
            return 10 + round(n / total * 75) if total else 85

        for start in range(0, total, pages_per_call):
            batch = pages[start : start + pages_per_call]
            end = min(start + len(batch), total)
            _report(progress, page_pct(start), f"Processing page {start + 1} of {total}...")
            t0 = _time.perf_counter()
            outputs, n_tokens = model.infer_multi(
                tokenizer,
                prompt="<image>Multi page parsing.",
                image_files=batch,
                output_path=work,
                image_size=1024,
                max_length=MAX_LENGTH,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
                ngram_window=NGRAM_WINDOW_MULTI,
                save_results=False,
            )
            gen_seconds += _time.perf_counter() - t0
            total_tokens += int(n_tokens or 0)
            raw_chunks = (outputs or "").split("<PAGE>")
            if len(raw_chunks) != len(batch):
                # Model didn't return exactly one chunk per page; keep the
                # non-empty ones and number them sequentially.
                raw_chunks = [c for c in raw_chunks if c.strip()]
            for j, raw_page in enumerate(raw_chunks):
                page_records.append(_page_record(start + j + 1, raw_page))
            _report(progress, page_pct(end), f"Processing page {end} of {total}...")

    _report(progress, 90, "Assembling output...")
    markdown = "\n\n---\n\n".join(p["markdown"] for p in page_records if p["markdown"])
    return {
        "markdown": markdown,
        "page_count": total,
        "dpi": dpi,
        "pages": page_records,
        "stats": _gen_stats(total_tokens, gen_seconds),   # model generation only (excludes rasterising)
    }


def ocr_pdf(
    model,
    tokenizer,
    pdf_path: str,
    dpi: int = PDF_DPI,
    pages_per_call: int = PAGES_PER_CALL,
    progress=None,
) -> str:
    """Markdown-only convenience wrapper around `ocr_pdf_detailed()`."""
    return ocr_pdf_detailed(model, tokenizer, pdf_path, dpi=dpi, pages_per_call=pages_per_call, progress=progress)["markdown"]
