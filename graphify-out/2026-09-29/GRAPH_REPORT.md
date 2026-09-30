# Graph Report - markitdown-app  (2026-09-23)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 259 nodes · 446 edges · 8 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.85)
- Token cost: 59,697 input · 121 output

## Graph Freshness
- Built from commit: `8fe46aff`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- OCR PDF Extraction
- File Conversion Dispatch
- Streamlit App UI
- Output Formatting Rendering
- Audio Whisper Transcription
- Ollama Model Selection
- URL Content Fetching
- HTML Parsing Helpers

## God Nodes (most connected - your core abstractions)
1. `dispatch()` - 17 edges
2. `ConversionResult` - 13 edges
3. `StageProgress` - 12 edges
4. `convert_document()` - 12 edges
5. `convert_image()` - 12 edges
6. `LocalModel` - 10 edges
7. `convert_media()` - 10 edges
8. `convert_url()` - 10 edges
9. `_HTMLTextExtractor` - 9 edges
10. `ocr_pdf_detailed()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `convert_media()` --references--> `ConversionResult`  [EXTRACTED]
  markitdown_app.py → file_handlers.py
- `convert_document()` --calls--> `build_sidecar()`  [EXTRACTED]
  markitdown_app.py → unlimited_ocr.py
- `convert_image()` --calls--> `_gen_stats()`  [EXTRACTED]
  markitdown_app.py → unlimited_ocr.py
- `get_unlimited_ocr_model()` --calls--> `load_model()`  [EXTRACTED]
  markitdown_app.py → unlimited_ocr.py
- `convert_image()` --calls--> `ocr_image_detailed()`  [EXTRACTED]
  markitdown_app.py → unlimited_ocr.py

## Import Cycles
- None detected.

## Communities (8 total, 0 thin omitted)

### Community 0 - "OCR PDF Extraction"
Cohesion: 0.05
Nodes (48): ast, tempfile, threading, time, _block_type(), build_sidecar(), clean_output(), extract_tables() (+40 more)

### Community 1 - "File Conversion Dispatch"
Cohesion: 0.08
Nodes (47): asyncio, classify(), ConversionResult, convert_archive(), convert_email(), convert_pptx(), walk(), convert_spreadsheet() (+39 more)

### Community 2 - "Streamlit App UI"
Cohesion: 0.07
Nodes (25): base64, pdf_page_count(), pdf_page_png(), Number of pages in a PDF given its bytes (PyMuPDF)., Rasterise one page (0-based) of a PDF to PNG bytes with PyMuPDF. Used for the…, get_unlimited_ocr_model(), human_size(), pdf_page_cached() (+17 more)

### Community 3 - "Output Formatting Rendering"
Cohesion: 0.10
Nodes (30): datetime, html, html_parser, io, render_cached(), filename_for(), _find_font_set(), _inline_runs() (+22 more)

### Community 4 - "Audio Whisper Transcription"
Cohesion: 0.10
Nodes (26): cache_resource, convert_media(), get_whisper_model(), Load a faster-whisper model once per size (first call downloads the weights)., os, _add_cuda_dll_dirs(), availability(), cublas_available() (+18 more)

### Community 5 - "Ollama Model Selection"
Cohesion: 0.09
Nodes (21): cache_data, dataclasses, json, _local_ollama_models(), ollama_model_picker(), Caption for the selected model + an expander listing every model's strengths., Selectbox over the Ollama models installed on this machine (text or vision),…, render_model_notes() (+13 more)

### Community 6 - "URL Content Fetching"
Cohesion: 0.10
Nodes (17): Exception, convert_url(), extract_youtube_id(), fetch_url_content(), fetch_with_browser(), fetch_youtube_transcript(), _friendly_url_error(), _HTMLTextExtractor (+9 more)

### Community 7 - "HTML Parsing Helpers"
Cohesion: 0.18
Nodes (3): HTMLParser, _TableHTMLParser, _TextExtractor

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_HTMLTextExtractor` connect `URL Content Fetching` to `File Conversion Dispatch`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Should `OCR PDF Extraction` be split into smaller, more focused modules?**
  _Cohesion score 0.050505050505050504 - nodes in this community are weakly interconnected._
- **Should `File Conversion Dispatch` be split into smaller, more focused modules?**
  _Cohesion score 0.08067375886524823 - nodes in this community are weakly interconnected._
- **Should `Streamlit App UI` be split into smaller, more focused modules?**
  _Cohesion score 0.07196969696969698 - nodes in this community are weakly interconnected._
- **Should `Output Formatting Rendering` be split into smaller, more focused modules?**
  _Cohesion score 0.10483870967741936 - nodes in this community are weakly interconnected._
- **Should `Audio Whisper Transcription` be split into smaller, more focused modules?**
  _Cohesion score 0.10098522167487685 - nodes in this community are weakly interconnected._
- **Should `Ollama Model Selection` be split into smaller, more focused modules?**
  _Cohesion score 0.09116809116809117 - nodes in this community are weakly interconnected._