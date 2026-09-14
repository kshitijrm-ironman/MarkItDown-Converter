"""
faster-whisper transcription for audio and video files.

Runs on CUDA (float16) when a GPU is visible to CTranslate2 — the same GPU
Unlimited-OCR uses — and falls back to CPU (int8) otherwise. Audio is decoded
with PyAV (bundled with faster-whisper), so video containers (mp4/mkv/avi/mov)
work without an external ffmpeg binary: the audio track is decoded straight
from the container.

Nothing heavy is imported at module import time; call `availability()` first.
"""

from __future__ import annotations

import os

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "distil-large-v3"]
DEFAULT_MODEL = "medium"
SAMPLE_RATE = 16000

_MODEL_NOTES = {
    "tiny": "~75 MB, fastest, lowest accuracy",
    "base": "~140 MB",
    "small": "~460 MB",
    "medium": "~1.5 GB, good default",
    "large-v3": "~3 GB, best accuracy",
    "large-v3-turbo": "~1.6 GB, near large-v3 accuracy, much faster",
    "distil-large-v3": "~1.5 GB, English-only, fast",
}


MODEL_BEST_AT = {
    "tiny":            "Best at: instant rough drafts of clear English speech; previews and timestamp checks. "
                       "~75 MB — will mis-hear accents, names and noisy audio.",
    "base":            "Best at: short, clean voice notes where speed matters more than every word. ~140 MB.",
    "small":           "Best at: the speed/accuracy sweet spot for everyday English recordings and meetings. ~460 MB.",
    "medium":          "Best at: multilingual audio, accents and moderate background noise — the safe default. ~1.5 GB.",
    "large-v3":        "Best at: maximum accuracy — technical vocabulary, non-English languages, poor-quality or "
                       "overlapping audio. ~3 GB, slowest.",
    "large-v3-turbo":  "Best at: near large-v3 accuracy at roughly 6× the speed — the best pick when a GPU is available. ~1.6 GB.",
    "distil-large-v3": "Best at: long English-only files (lectures, podcasts) — very fast, large-class accuracy. "
                       "~1.5 GB; English only.",
}


def model_label(size: str) -> str:
    return f"{size}  ({_MODEL_NOTES.get(size, '')})"


# --------------------------------------------------------------------------- #
# CUDA runtime plumbing
# --------------------------------------------------------------------------- #
_CUBLAS_DLL = "cublas64_12.dll"  # cuDNN ships inside the ctranslate2 wheel; cuBLAS does not.


def _add_cuda_dll_dirs() -> list[str]:
    """
    CTranslate2 needs cuBLAS / cuDNN DLLs on Windows. The torch CUDA wheel
    (installed by requirements-ocr.txt) ships them in torch/lib; the nvidia-*
    pip packages ship them under nvidia/<lib>/bin. Register whichever exist
    and return the directories that were registered.
    """
    candidates: list[str] = []
    try:
        import torch  # noqa: F401

        candidates.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception:
        pass
    try:
        import nvidia  # type: ignore

        base = os.path.dirname(nvidia.__file__)
        for sub in ("cublas", "cudnn", "cuda_runtime"):
            candidates.append(os.path.join(base, sub, "bin"))
    except Exception:
        pass
    registered: list[str] = []
    for path in candidates:
        if os.path.isdir(path):
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(path)
            except Exception:
                pass
            if path not in os.environ.get("PATH", ""):
                os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            registered.append(path)
    return registered


def cuda_device_count() -> int:
    try:
        import ctranslate2

        _add_cuda_dll_dirs()
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def cublas_available() -> bool:
    """
    A visible CUDA device is not enough: CTranslate2 loads cublas64_12.dll
    lazily at model-load time and falls over if it is missing. Check the
    registered DLL dirs and PATH so the UI can say *why* it is on CPU.
    """
    if os.name != "nt":
        return True
    dirs = _add_cuda_dll_dirs() + os.environ.get("PATH", "").split(os.pathsep)
    return any(d and os.path.exists(os.path.join(d, _CUBLAS_DLL)) for d in dirs)


def gpu_ready() -> tuple[bool, str]:
    """(usable, reason) for GPU transcription."""
    n = cuda_device_count()
    if n == 0:
        return False, "No CUDA GPU visible to CTranslate2"
    if not cublas_available():
        return False, (
            f"CUDA GPU detected but the cuBLAS runtime ({_CUBLAS_DLL}) is missing — "
            "install the GPU extras (Run MarkItDown.bat → GPU extras, or pip install -r requirements-ocr.txt)"
        )
    return True, f"CUDA GPU detected ({n} device{'s' if n > 1 else ''})"


def availability() -> tuple[bool, str]:
    """Return (installed, detail). Transcription works on CPU too, just slower."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False, (
            "faster-whisper is not installed. Run 'Run MarkItDown.bat' and accept the "
            "GPU extras, or: pip install faster-whisper"
        )
    ok, reason = gpu_ready()
    if ok:
        return True, f"{reason} — float16 on GPU."
    return True, f"{reason}. Transcribing on CPU (int8, slower)."


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(size: str = DEFAULT_MODEL):
    """
    Load a faster-whisper model. Returns (model, device) where device is
    "cuda" or "cpu". Expensive on first call (weights are downloaded from
    Hugging Face and cached); cache the result in the app.

    GPU loading is verified with a tiny warm-up so a missing cuBLAS/cuDNN
    DLL surfaces here (and triggers the CPU fallback) instead of mid-file.
    """
    import numpy as np
    from faster_whisper import WhisperModel

    if size not in MODEL_SIZES:
        raise ValueError(f"Unknown model size {size!r}; expected one of {MODEL_SIZES}")

    if gpu_ready()[0]:
        try:
            model = WhisperModel(size, device="cuda", compute_type="float16")
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            list(model.transcribe(silence, beam_size=1)[0])
            return model, "cuda"
        except Exception:
            pass  # fall through to CPU
    model = WhisperModel(size, device="cpu", compute_type="int8")
    return model, "cpu"


# --------------------------------------------------------------------------- #
# Decoding / transcription
# --------------------------------------------------------------------------- #
def extract_audio(media_path: str):
    """
    Decode the audio track of any audio *or video* container into 16 kHz mono
    float32 samples using PyAV. This is the "extract audio" step for videos —
    no intermediate file and no ffmpeg binary required.
    """
    from faster_whisper import decode_audio

    return decode_audio(media_path, sampling_rate=SAMPLE_RATE)


def transcribe(
    model,
    media,
    language: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    progress=None,
) -> dict:
    """
    Transcribe an audio/video file.

    `media` is either a path (decoded here with `extract_audio`) or the
    float32 sample array `extract_audio` returned (lets the caller report the
    extraction as its own stage).

    Returns {"language", "language_probability", "duration", "text", "segments"}
    where segments = [{"start", "end", "text"}, ...] (seconds).

    `progress(done_seconds, total_seconds, n_segments)` is called every time
    faster-whisper's `transcribe()` generator yields a segment — decoding is
    lazy, so this reflects real-time transcription progress.
    """
    audio = extract_audio(media) if isinstance(media, (str, os.PathLike)) else media
    duration = float(len(audio)) / SAMPLE_RATE
    segments_iter, info = model.transcribe(
        audio,
        language=language or None,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )
    segments: list[dict] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({"start": round(float(seg.start), 2), "end": round(float(seg.end), 2), "text": text})
        if progress and duration > 0:
            progress(min(float(seg.end), duration), duration, len(segments))
    return {
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "duration": round(duration, 2),
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
    }


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _paragraphs(segments: list[dict], gap: float = 2.0, max_chars: int = 700) -> list[str]:
    """Group consecutive segments into readable paragraphs."""
    paras: list[str] = []
    current: list[str] = []
    last_end = None
    for seg in segments:
        new_para = (
            current
            and (
                (last_end is not None and seg["start"] - last_end > gap)
                or sum(len(t) + 1 for t in current) > max_chars
            )
        )
        if new_para:
            paras.append(" ".join(current))
            current = []
        current.append(seg["text"])
        last_end = seg["end"]
    if current:
        paras.append(" ".join(current))
    return paras


def to_markdown(result: dict, source_name: str, kind: str, model_size: str, device: str) -> str:
    """Render a transcription result as Markdown (paragraphs + timestamped segments)."""
    lines = [f"# Transcript: {source_name}", ""]
    lines.append(
        f"> Transcribed by faster-whisper `{model_size}` on {device.upper()} · "
        f"language: {result['language']} ({result['language_probability']:.0%}) · "
        f"duration: {format_timestamp(result['duration'])}"
    )
    if kind == "video":
        lines.append("> Audio track extracted from the video container.")
    lines.append("")
    if not result["segments"]:
        lines.append("_No speech detected._")
        return "\n".join(lines)
    lines.append("## Transcript")
    lines.append("")
    for para in _paragraphs(result["segments"]):
        lines.append(para)
        lines.append("")
    lines.append("## Timestamped segments")
    lines.append("")
    for seg in result["segments"]:
        lines.append(f"- **[{format_timestamp(seg['start'])} → {format_timestamp(seg['end'])}]** {seg['text']}")
    return "\n".join(lines).strip() + "\n"
