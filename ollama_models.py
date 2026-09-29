"""
Discover the Ollama models that are actually installed on this machine.

Talks to the local Ollama server (GET /api/tags) so the UI can offer the
models you already have instead of a hard-coded list. Pure module — no
Streamlit imports; the app caches the result.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST

# Families that are vision-capable even when the server does not report
# `capabilities` (older Ollama versions).
_VISION_FAMILY_HINTS = ("llava", "vision", "vl", "moondream", "clip", "minicpm-v", "bakllava",
                        "gemma3", "pixtral", "internvl")


# Where a previously discovered model store is remembered, so the disk scan
# happens once per machine instead of on every app start.
_STORE_CACHE = Path.home() / ".markitdown" / "ollama_store.json"


def _is_model_store(path: Path) -> bool:
    """A real Ollama store has blobs/sha256-* files and manifests/."""
    try:
        blobs = path / "blobs"
        if not blobs.is_dir() or not (path / "manifests").is_dir():
            return False
        return any(blobs.glob("sha256-*"))
    except OSError:
        return False


def _cached_store() -> Path | None:
    try:
        p = Path(json.loads(_STORE_CACHE.read_text())["path"])
    except Exception:  # noqa: BLE001
        return None
    return p if _is_model_store(p) else None


def _remember_store(path: Path) -> None:
    try:
        _STORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _STORE_CACHE.write_text(json.dumps({"path": str(path)}))
    except OSError:
        pass


def _drive_roots() -> list[Path]:
    if sys.platform == "win32":
        return [Path(f"{c}:/") for c in "CDEFGHIJKLMNOPQRSTUVWXYZ" if Path(f"{c}:/").is_dir()]
    return [Path.home(), Path("/mnt"), Path("/media"), Path("/Volumes")]


def _scan_for_store(max_depth: int = 6) -> Path | None:
    """Walk the machine's drives for a directory holding blobs/ + manifests/.

    Breadth-first with a depth cap so a 1 TB disk does not take minutes; the
    result is cached, so this runs at most once per machine.
    """
    skip = {"windows", "$recycle.bin", "system volume information", "node_modules",
            ".git", "appdata", "winsxs", "program files", "program files (x86)"}
    for root in _drive_roots():
        queue: list[tuple[Path, int]] = [(root, 0)]
        while queue:
            current, depth = queue.pop(0)
            if _is_model_store(current):
                return current
            if depth >= max_depth:
                continue
            try:
                children = [d for d in current.iterdir() if d.is_dir()]
            except OSError:
                continue
            for child in children:
                if child.name.lower() in skip or child.name.startswith("$"):
                    continue
                queue.append((child, depth + 1))
    return None


def find_model_store() -> Path | None:
    """Locate this machine's Ollama model store, wherever it lives.

    Order: OLLAMA_MODELS → the default ~/.ollama/models (only when it really
    holds blobs) → remembered location → full disk scan.
    """
    env = os.environ.get("OLLAMA_MODELS")
    if env and _is_model_store(Path(env)):
        return Path(env)

    default = Path.home() / ".ollama" / "models"
    if _is_model_store(default):
        return default

    cached = _cached_store()
    if cached:
        return cached

    found = _scan_for_store()
    if found:
        _remember_store(found)
    return found


def find_ollama_binary() -> str | None:
    """Locate the `ollama` executable on this machine — PATH first, then the
    per-user / system install locations each OS installer actually uses."""
    found = shutil.which("ollama")
    if found:
        return found

    home = Path.home()
    if sys.platform == "win32":
        candidates = [
            home / "AppData/Local/Programs/Ollama/ollama.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe",
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Ollama/ollama.exe",
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Ollama/ollama.exe",
        ]
    elif sys.platform == "darwin":
        candidates = [
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
            Path("/usr/local/bin/ollama"),
            Path("/opt/homebrew/bin/ollama"),
            home / ".ollama/bin/ollama",
        ]
    else:
        candidates = [
            Path("/usr/local/bin/ollama"),
            Path("/usr/bin/ollama"),
            Path("/snap/bin/ollama"),
            home / ".local/bin/ollama",
        ]
    for c in candidates:
        try:
            if c.is_file():
                return str(c)
        except OSError:
            continue
    return None


def _server_alive(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def _stop_server() -> None:
    """Best-effort shutdown of a running `ollama serve` (so it can be restarted
    pointing at the store that actually holds this machine's models)."""
    cmd = (["taskkill", "/F", "/IM", "ollama.exe"] if sys.platform == "win32"
           else ["pkill", "-f", "ollama serve"])
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(1.5)


def ensure_server(wait: float = 20.0, restart: bool = False) -> str:
    """Start `ollama serve` in the background if it is not already answering.

    The server is pointed at this machine's real model store via OLLAMA_MODELS,
    so the models are found wherever they live (another drive, another user
    profile). Returns "" on success, otherwise a human-readable reason.
    """
    if _server_alive() and not restart:
        return ""
    if restart:
        _stop_server()

    exe = find_ollama_binary()
    if not exe:
        return ("Ollama is not installed on this machine (no `ollama` executable on PATH "
                "or in the default install folder). Install it from https://ollama.com/download.")

    env = os.environ.copy()
    store = find_model_store()
    if store:
        env["OLLAMA_MODELS"] = str(store)

    kwargs: dict = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL, env=env)
    if sys.platform == "win32":
        kwargs["creationflags"] = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                                   | getattr(subprocess, "DETACHED_PROCESS", 0))
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen([exe, "serve"], **kwargs)
    except OSError as exc:
        return f"Could not start Ollama ({exe}): {exc}"

    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if _server_alive():
            return ""
        time.sleep(0.5)
    return f"Started `{exe} serve` but the server did not answer on {OLLAMA_HOST} within {wait:.0f}s."


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
        # /api/tags does not report `capabilities`, and some builds report an
        # incomplete list, so the name/family hints are always consulted too —
        # never trust an empty/partial capability list to mean "no vision".
        if "vision" in self.capabilities:
            return True
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
    except urllib.error.URLError:
        # Ollama is installed but not serving on this machine — start it and retry.
        started = ensure_server()
        if started:
            return [], started
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.load(resp)
        except Exception as exc:  # noqa: BLE001
            return [], f"Ollama server not reachable at {OLLAMA_HOST} ({exc}). Start it with `ollama serve`."
    except Exception as exc:  # noqa: BLE001
        return [], f"Could not read the Ollama model list: {exc}"

    if not data.get("models"):
        # Server is up but sees nothing: it was launched against the empty
        # default folder while the models sit in another store. Point it at the
        # real one and ask again.
        store = find_model_store()
        if store and str(store) != os.environ.get("_MD_TRIED_STORE", ""):
            os.environ["_MD_TRIED_STORE"] = str(store)
            if not ensure_server(restart=True):
                try:
                    with urllib.request.urlopen(url, timeout=timeout) as resp:
                        data = json.load(resp)
                except Exception:  # noqa: BLE001
                    pass

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
