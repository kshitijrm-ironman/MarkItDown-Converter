"""Unit tests for the pure helpers — no Streamlit runtime, no model, no I/O.

Run:  venv/Scripts/python -m pytest tests -q
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import file_handlers as fh          # noqa: E402
import unlimited_ocr                # noqa: E402

ext_of = fh.ext_of
classify = fh.classify
join_pages = fh.join_pages
clean_output = unlimited_ocr.clean_output
STOP_TOKEN = unlimited_ocr.STOP_TOKEN


# --------------------------------------------------------------------------- #
# ext_of
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name, expected",
    [
        ("report.pdf", ".pdf"),
        ("REPORT.PDF", ".pdf"),          # lowercased
        ("photo.JpEg", ".jpeg"),
        ("README", ""),                  # no extension
        ("", ""),                        # empty string
        (None, ""),                      # `name or ""` guard
        (".gitignore", ""),              # dotfile: all stem, no ext
        (".env.local", ".local"),        # dotfile that does have one
        ("archive.tar.gz", ".gz"),       # compound: last segment only
        ("scan.v2.final.tiff", ".tiff"),
        ("dir.with.dots/file", ""),
        ("C:\\tmp\\deck.PPTX", ".pptx"),
        ("trailing.", "."),              # splitext keeps the bare dot
    ],
)
def test_ext_of(name, expected):
    assert ext_of(name) == expected


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name, kind",
    [
        ("a.png", "image"),
        ("a.heic", "image"),
        ("a.TIFF", "image"),             # case-insensitive via ext_of
        ("a.mp3", "audio"),
        ("a.flac", "audio"),
        ("a.mp4", "video"),
        ("a.mkv", "video"),
        ("a.srt", "subtitle"),
        ("a.vtt", "subtitle"),
        ("a.pptx", "presentation"),
        ("a.xlsx", "spreadsheet"),
        ("a.csv", "spreadsheet"),
        ("a.eml", "email"),
        ("a.zip", "archive"),
        ("a.pdf", "document"),           # documents are the fallback bucket
        ("a.docx", "document"),
        ("a.xyz", "document"),           # unsupported type -> document, never raises
        ("no_extension", "document"),
        ("", "document"),
    ],
)
def test_classify(name, kind):
    assert classify(name) == kind


@pytest.mark.parametrize(
    "exts, kind",
    [
        (fh.IMAGE_EXTS, "image"),
        (fh.AUDIO_EXTS, "audio"),
        (fh.VIDEO_EXTS, "video"),
        (fh.SUBTITLE_EXTS, "subtitle"),
        (fh.PRESENTATION_EXTS, "presentation"),
        (fh.SPREADSHEET_EXTS, "spreadsheet"),
        (fh.EMAIL_EXTS, "email"),
        (fh.ARCHIVE_EXTS, "archive"),
    ],
)
def test_classify_covers_every_declared_ext(exts, kind):
    """Every extension in each set maps to its own kind — no gaps, no overlap."""
    for ext in exts:
        assert classify(f"sample{ext}") == kind


def test_classify_returns_a_declared_conversionresult_kind():
    kinds = {"document", "image", "audio", "video", "subtitle",
             "presentation", "spreadsheet", "email", "archive", "url"}
    assert classify("whatever.bin") in kinds


# --------------------------------------------------------------------------- #
# join_pages
# --------------------------------------------------------------------------- #
def test_join_pages_empty_list():
    out = join_pages([], "scan.pdf", "Extracted Text")
    assert out == "# Extracted Text\n\n> Source file: scan.pdf\n"
    assert "## Page" not in out


def test_join_pages_single_page_has_no_page_heading():
    out = join_pages(["Hello world"], "scan.png", "Extracted Text")
    assert out == "# Extracted Text\n\n> Source file: scan.png\n\nHello world"
    assert "## Page" not in out
    assert "---" not in out


def test_join_pages_single_empty_page_gets_placeholder():
    out = join_pages([""], "blank.png", "Extracted Text")
    assert out.endswith("_No text found in this image._")


def test_join_pages_multi_page_numbers_and_separates():
    out = join_pages(["one", "two", "three"], "doc.pdf", "Extracted Text")
    assert "## Page 1\n\none" in out
    assert "## Page 2\n\ntwo" in out
    assert "## Page 3\n\nthree" in out
    assert out.count("---\n\n") == 2          # separators between, not after last
    assert not out.rstrip("\n").endswith("---")


def test_join_pages_multi_page_empty_page_gets_placeholder():
    out = join_pages(["text", ""], "doc.pdf", "Extracted Text")
    assert "_No text found on this page._" in out


def test_join_pages_strips_trailing_newlines_to_exactly_one():
    out = join_pages(["one\n\n\n", "two\n\n\n"], "doc.pdf", "Extracted Text")
    assert out.endswith("two\n")
    assert not out.endswith("two\n\n")


def test_join_pages_byline_is_inserted_after_source_line():
    out = join_pages(["body"], "x.png", "Image Analysis", "Described by Claude AI")
    assert out.startswith(
        "# Image Analysis\n\n> Source file: x.png\n\n> Described by Claude AI\n\n"
    )


def test_join_pages_without_byline_emits_no_second_quote():
    out = join_pages(["body"], "x.png", "Extracted Text")
    assert out.count("> ") == 1


def test_join_pages_handles_none_pages():
    out = join_pages([None, None], "doc.pdf", "Extracted Text")
    assert out.count("_No text found on this page._") == 2


# --------------------------------------------------------------------------- #
# clean_output
# --------------------------------------------------------------------------- #
def test_clean_output_empty_string():
    assert clean_output("") == ""


def test_clean_output_none_is_falsy_guarded():
    assert clean_output(None) == ""


def test_clean_output_whitespace_only():
    assert clean_output("   \n\n\t  \n ") == ""


def test_clean_output_collapses_triple_newlines():
    assert clean_output("a\n\n\nb") == "a\n\nb"
    assert clean_output("a\n\n\n\n\n\nb") == "a\n\nb"


def test_clean_output_preserves_single_blank_line():
    assert clean_output("a\n\nb") == "a\n\nb"


def test_clean_output_strips_stop_token():
    assert clean_output(f"done{STOP_TOKEN}") == "done"
    assert STOP_TOKEN not in clean_output(f"{STOP_TOKEN}text{STOP_TOKEN}")


def test_clean_output_strips_bare_det_markers():
    out = clean_output("Title<|det|>[[10, 20, 30, 40]]<|/det|>\nBody")
    assert out == "Title\nBody"


def test_clean_output_ref_image_becomes_placeholder():
    out = clean_output("<|ref|>image<|/ref|><|det|>[[0, 0, 9, 9]]<|/det|>")
    assert out == "*[image]*"


def test_clean_output_ref_non_image_is_dropped_entirely():
    out = clean_output("<|ref|>title<|/ref|><|det|>[[0, 0, 9, 9]]<|/det|>Heading")
    assert out == "Heading"


def test_clean_output_ref_label_match_is_case_and_space_insensitive():
    assert clean_output("<|ref|>  IMAGE  <|/ref|> <|det|>[[1,2,3,4]]<|/det|>") == "*[image]*"


def test_clean_output_rewrites_latex_colon_operators():
    assert clean_output(r"x \coloneqq y") == "x := y"
    assert clean_output(r"y \eqqcolon x") == "y =: x"


def test_clean_output_rstrips_each_line():
    assert clean_output("alpha   \nbeta\t\ngamma  ") == "alpha\nbeta\ngamma"


def test_clean_output_strips_leading_and_trailing_blank_lines():
    assert clean_output("\n\n# Heading\n\ntext\n\n\n") == "# Heading\n\ntext"


def test_clean_output_is_idempotent():
    raw = f"\n\n<|ref|>image<|/ref|><|det|>[[1,2,3,4]]<|/det|>\n\n\nTail  {STOP_TOKEN}\n"
    once = clean_output(raw)
    assert clean_output(once) == once


def test_clean_output_leaves_plain_markdown_untouched():
    md = "# Title\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n- bullet"
    assert clean_output(md) == md
