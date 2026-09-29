# Graph Report - markitdown-app  (2026-09-29)

## Corpus Check
- 11 files · ~18,474 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: .bat 5, (none) 3, .toml 1)

## Summary
- 329 nodes · 548 edges · 12 communities (11 shown, 1 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8fe46aff`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- unlimited_ocr.py
- file_handlers.py
- markitdown_app.py
- output_formatter.py
- whisper_handler.py
- MarkItDown App — Document → Markdown converter (UI by Kshitij)
- _HTMLTextExtractor
- ollama_models.py
- convert_document
- doc_password.py
- os
- CLAUDE.md

## God Nodes (most connected - your core abstractions)
1. `dispatch()` - 18 edges
2. `MarkItDown App — Document → Markdown converter (UI by Kshitij)` - 16 edges
3. `ConversionResult` - 13 edges
4. `StageProgress` - 12 edges
5. `convert_image()` - 12 edges
6. `convert_document()` - 12 edges
7. `convert_url()` - 10 edges
8. `convert_media()` - 10 edges
9. `LocalModel` - 10 edges
10. `ext_of()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `convert_archive()` --calls--> `PasswordRequired`  [EXTRACTED]
  file_handlers.py → doc_password.py
- `dispatch()` --calls--> `unlock_in_place()`  [EXTRACTED]
  markitdown_app.py → doc_password.py
- `convert_document()` --references--> `ConversionResult`  [EXTRACTED]
  markitdown_app.py → file_handlers.py
- `convert_image()` --references--> `ConversionResult`  [EXTRACTED]
  markitdown_app.py → file_handlers.py
- `convert_media()` --references--> `ConversionResult`  [EXTRACTED]
  markitdown_app.py → file_handlers.py

## Import Cycles
- None detected.

## Communities (12 total, 1 thin omitted)

### Community 0 - "unlimited_ocr.py"
Cohesion: 0.05
Nodes (45): ast, tempfile, threading, _block_type(), clean_output(), extract_tables(), _from_pretrained(), _gen_stats() (+37 more)

### Community 1 - "file_handlers.py"
Cohesion: 0.07
Nodes (51): asyncio, classify(), ConversionResult, convert_archive(), convert_email(), convert_pptx(), walk(), convert_spreadsheet() (+43 more)

### Community 2 - "markitdown_app.py"
Cohesion: 0.08
Nodes (25): base64, pdf_page_count(), pdf_page_png(), Number of pages in a PDF given its bytes (PyMuPDF)., Rasterise one page (0-based) of a PDF to PNG bytes with PyMuPDF. Used for the…, human_size(), pdf_page_cached(), Exception (+17 more)

### Community 3 - "output_formatter.py"
Cohesion: 0.07
Nodes (33): datetime, html, html_parser, io, render_cached(), filename_for(), _find_font_set(), _inline_runs() (+25 more)

### Community 4 - "whisper_handler.py"
Cohesion: 0.11
Nodes (25): cache_resource, convert_media(), get_whisper_model(), Load a faster-whisper model once per size (first call downloads the weights)., _add_cuda_dll_dirs(), availability(), cublas_available(), cuda_device_count() (+17 more)

### Community 5 - "MarkItDown App — Document → Markdown converter (UI by Kshitij)"
Cohesion: 0.05
Nodes (35): cache_data, _local_ollama_models(), ollama_model_picker(), Caption for the selected model + an expander listing every model's strengths., Selectbox over the Ollama models installed on this machine (text or vision),…, render_model_notes(), LocalModel, note_for() (+27 more)

### Community 6 - "_HTMLTextExtractor"
Cohesion: 0.22
Nodes (3): _HTMLTextExtractor, HTMLParser, Very small HTML → text fallback (used only if MarkItDown is unavailable).

### Community 7 - "ollama_models.py"
Cohesion: 0.11
Nodes (29): dataclasses, json, _cached_store(), _drive_roots(), ensure_server(), find_model_store(), find_ollama_binary(), _is_model_store() (+21 more)

### Community 8 - "convert_document"
Cohesion: 0.13
Nodes (16): convert_with_markitdown(), join_pages(), ocr_pages_tesseract(), Combine per-page Markdown into one document (page headings only when > 1 page)., convert_document(), convert_image(), get_unlimited_ocr_model(), _ocr_fallback() (+8 more)

### Community 9 - "doc_password.py"
Cohesion: 0.27
Nodes (9): _head(), PasswordRequired, Exception, Password-protected documents: detect them, and unlock them when the user…, A file is encrypted and no password (or a wrong one) was supplied., Decrypt `path` in place when it is an encrypted PDF or Office file. Returns…, unlock_in_place(), _unlock_office() (+1 more)

### Community 10 - "os"
Cohesion: 0.25
Nodes (9): Image, os, pil, build(), _gradient(), _lerp(), main(), Draw the MarkItDown app logo and write assets/logo.png + assets/markitdown.ico.… (+1 more)

## Knowledge Gaps
- **18 isolated node(s):** `graphify`, `Contents`, `1. Features at a glance`, `Mandatory`, `Optional — enable individual features` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 159 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LocalModel` connect `MarkItDown App — Document → Markdown converter (UI by Kshitij)` to `ollama_models.py`?**
  _High betweenness centrality (0.169) - this node is a cross-community bridge._
- **What connects `graphify`, `Contents`, `1. Features at a glance` to the rest of the system?**
  _18 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `unlimited_ocr.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05429864253393665 - nodes in this community are weakly interconnected._
- **Should `file_handlers.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07164404223227752 - nodes in this community are weakly interconnected._
- **Should `markitdown_app.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07586206896551724 - nodes in this community are weakly interconnected._
- **Should `output_formatter.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07188160676532769 - nodes in this community are weakly interconnected._
- **Should `whisper_handler.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10582010582010581 - nodes in this community are weakly interconnected._