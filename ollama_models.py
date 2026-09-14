"""
Discover the Ollama models that are actually installed on this machine.

Talks to the local Ollama server (GET /api/tags) so the UI can offer the
models you already have instead of a hard-coded list. Pure module — no
Streamlit imports; the app caches the result.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST

# Families that are vision-capable even when the server does not report
# `capabilities` (older Ollama versions).
_VISION_FAMILY_HINTS = ("llava", "vision", "vl", "moondream", "clip", "minicpm-v", "bakllava", "gemma3")


@dataclass
class LocalModel:
    name: str                         # exact tag to pass to ollama.chat(), e.g. "qwen2.5vl:7b"
    family: str = ""
    params: str = ""                  # "14.8B"
    quant: str = ""                   # "Q4_K_M"
    context: int | None = None        # tokens
    size_gb: float = 0.0
    capabilities: list[str] = field(default_factory=list)

    @property
    def base(self) -> str:
        """Name without the tag: 'qwen2.5vl:7b' -> 'qwen2.5vl'."""
        return self.name.split(":", 1)[0]

    @property
    def vision(self) -> bool:
        if self.capabilities:
            return "vision" in self.capabilities
        probe = f"{self.family} {self.base}".lower()
        return any(h in probe for h in _VISION_FAMILY_HINTS)

    @property
    def thinking(self) -> bool:
        return "thinking" in self.capabilities or self.base.startswith("deepseek-r1")

    def summary(self) -> str:
        """Short hardware-oriented description: '14.8B · Q4_K_M · 9.0 GB · 32k context'."""
        bits = []
        if self.params:
            bits.append(self.params)
        if self.quant and self.quant.lower() != "unknown":
            bits.append(self.quant)
        if self.size_gb:
            bits.append(f"{self.size_gb:.1f} GB")
        if self.context:
            bits.append(f"{self.context // 1000}k context")
        if self.family:
            bits.append(f"family {self.family}")
        return " · ".join(bits)

    def label(self) -> str:
        """Selectbox label: 'qwen2.5:14b  (14.8B · 9.0 GB)'."""
        extra = " · ".join(b for b in (self.params, f"{self.size_gb:.1f} GB" if self.size_gb else "") if b)
        return f"{self.name}  ({extra})" if extra else self.name


def list_local_models(timeout: float = 3.0) -> tuple[list[LocalModel], str]:
    """
    Return (models, error). `error` is "" when the Ollama server answered;
    otherwise a short human-readable reason and `models` is empty.
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.URLError as exc:
        return [], f"Ollama server not reachable at {OLLAMA_HOST} ({getattr(exc, 'reason', exc)}). Start it with `ollama serve`."
    except Exception as exc:  # noqa: BLE001
        return [], f"Could not read the Ollama model list: {exc}"

    models: list[LocalModel] = []
    for m in data.get("models", []):
        det = m.get("details", {}) or {}
        models.append(LocalModel(
            name=m.get("name") or m.get("model", ""),
            family=det.get("family", "") or "",
            params=det.get("parameter_size", "") or "",
            quant=det.get("quantization_level", "") or "",
            context=det.get("context_length"),
            size_gb=(m.get("size") or 0) / 1e9,
            capabilities=list(m.get("capabilities") or []),
        ))
    models.sort(key=lambda x: x.name.lower())
    return models, ""


def split_by_kind(models: list[LocalModel]) -> tuple[list[LocalModel], list[LocalModel]]:
    """(text_models, vision_models). Vision models can also do text, but the
    document picker only lists pure text models to keep the choice obvious."""
    vision = [m for m in models if m.vision]
    text = [m for m in models if not m.vision]
    return text, vision


def note_for(model: LocalModel, notes: dict[str, str]) -> str:
    """Curated 'best at' note for an installed model, matched by exact tag,
    then base name, then family; falls back to an auto-generated summary."""
    for key in (model.name, model.base, model.base.split("/")[-1]):
        if key in notes:
            return notes[key]
    for key, text in notes.items():
        if key and (model.base.startswith(key) or model.family == key):
            return text
    return f"Installed locally — {model.summary()}." if model.summary() else "Installed locally."
