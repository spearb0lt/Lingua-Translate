"""Shared plumbing for local (on-device) translation models.

Handles the three things every local seq2seq provider needs: picking a device
that actually fits, splitting long text into model-sized pieces, and keeping
loaded models under a VRAM budget.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Sentence boundaries including the Devanagari danda and CJK full stop.
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥。！？])\s+")


@dataclass(frozen=True)
class DeviceChoice:
    device: str        # "cuda" or "cpu"
    dtype: str         # "float16" or "float32"
    vram_gb: float     # 0.0 on CPU
    reason: str


def pick_device(prefer: str | None = None, min_free_gb: float = 1.2) -> DeviceChoice:
    """Choose where to run. Honours LINGUA_DEVICE, else auto-detects.

    We refuse CUDA when free VRAM is below ``min_free_gb`` -- on a 4 GB laptop
    GPU it is better to be slow on CPU than to die with an OOM mid-request.
    """
    want = (prefer or os.getenv("LINGUA_DEVICE") or "auto").lower()

    if want == "cpu":
        return DeviceChoice("cpu", "float32", 0.0, "forced by configuration")

    try:
        import torch
    except ImportError:
        return DeviceChoice("cpu", "float32", 0.0, "torch not installed")

    if not torch.cuda.is_available():
        if want == "cuda":
            log.warning("LINGUA_DEVICE=cuda but no CUDA device is visible")
        return DeviceChoice("cpu", "float32", 0.0, "no CUDA device available")

    free_b, total_b = torch.cuda.mem_get_info()
    free_gb = free_b / (1024 ** 3)
    total_gb = total_b / (1024 ** 3)
    name = torch.cuda.get_device_name(0)

    if want == "cuda":
        return DeviceChoice("cuda", "float16", free_gb, f"forced; {name}")
    if free_gb < min_free_gb:
        return DeviceChoice("cpu", "float32", free_gb,
                            f"{name} has only {free_gb:.1f} GB free of {total_gb:.1f} GB")
    return DeviceChoice("cuda", "float16", free_gb,
                        f"{name}, {free_gb:.1f} GB free of {total_gb:.1f} GB")


def split_sentences(text: str, max_chars: int = 400) -> list[str]:
    """Break text into model-sized pieces on sentence boundaries.

    Seq2seq NMT models degrade badly past a few hundred tokens, so we
    translate sentence by sentence and rejoin. Overlong single sentences are
    word-wrapped rather than truncated.
    """
    text = text.strip()
    if not text:
        return []

    out: list[str] = []
    for sentence in _SENT_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            out.append(sentence)
            continue
        buf = ""
        for word in sentence.split(" "):
            if len(buf) + len(word) + 1 > max_chars and buf:
                out.append(buf)
                buf = word
            else:
                buf = f"{buf} {word}" if buf else word
        if buf:
            out.append(buf)
    return out


class ModelCache:
    """Keeps at most ``capacity`` models resident, evicting least-recently-used.

    A 4 GB laptop GPU cannot hold every IndicTrans2 direction plus NLLB plus
    Whisper at once, so we cap how many stay loaded and free the rest.
    """

    def __init__(self, capacity: int | None = None):
        if capacity is None:
            capacity = int(os.getenv("LINGUA_MODEL_CACHE", "2"))
        self.capacity = max(1, capacity)
        self._entries: dict[str, object] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()

    def get(self, key: str):
        with self._lock:
            if key not in self._entries:
                return None
            self._order.remove(key)
            self._order.append(key)
            return self._entries[key]

    def put(self, key: str, value) -> None:
        with self._lock:
            if key in self._entries:
                self._order.remove(key)
            self._entries[key] = value
            self._order.append(key)
            while len(self._order) > self.capacity:
                self._evict(self._order[0])

    def _evict(self, key: str) -> None:
        log.info("evicting model %s to free memory", key)
        self._entries.pop(key, None)
        if key in self._order:
            self._order.remove(key)
        free_accelerator_memory()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._order.clear()
        free_accelerator_memory()

    def loaded(self) -> list[str]:
        with self._lock:
            return list(self._order)


def free_accelerator_memory() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def model_root() -> str:
    """Where model weights live. Override with LINGUA_MODEL_DIR."""
    return os.getenv("LINGUA_MODEL_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "lingua-models")
