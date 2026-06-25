#!/usr/bin/env python3
"""Optional Confucius4-TTS bridge for the local dubbing pipeline.

The model is loaded lazily because it is a CUDA-focused, zero-shot voice-cloning
engine.  Keep its checkout beside this project (``E:\\dub\\Confucius4-TTS``), or
set ``CONFUCIUS_TTS_REPO`` to another checkout.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CONFUCIUS_TTS_REPO", os.path.join(HERE, "..", "Confucius4-TTS"))
CONFIG = os.environ.get("CONFUCIUS_TTS_CONFIG", os.path.join(REPO, "config", "inference_config.yaml"))
VENDOR = os.path.join(HERE, "_vendor", "confucius")

# App language codes -> Confucius4-TTS language codes.
LANGS = {
    "zh-CN": "zh", "en": "en", "ja": "ja", "ko": "ko", "de": "de", "fr": "fr",
    "es": "es", "id": "id", "it": "it", "th": "th", "pt": "pt", "ru": "ru",
    "ms": "ms", "vi": "vi",
}
_state = {"tried": False, "model": None, "reason": ""}


def enabled() -> bool:
    return os.environ.get("CONFUCIUS_TTS_ENABLE", "").strip().lower() in {"1", "true", "yes", "on"}


def available() -> bool:
    return enabled() and os.path.isfile(CONFIG) and _load() is not None


def unavailable_reason() -> str:
    """Return the actionable reason from the latest availability check."""
    return str(_state.get("reason") or "Confucius4-TTS could not be initialized.")


def _load():
    if _state["tried"]:
        return _state["model"]
    _state["tried"] = True
    if not os.path.isfile(CONFIG):
        _state["reason"] = f"checkout/config not found at {CONFIG}"
        print(f"  [confucius] {_state['reason']}; using normal voice.", flush=True)
        return None
    try:
        # Keep the heavyweight inference-only packages separate from the desktop
        # app so their Transformers/tokenizers versions never replace a running UI.
        if os.path.isdir(VENDOR) and VENDOR not in sys.path:
            sys.path.append(VENDOR)
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        import torch
        from confuciustts.cli.inference import ConfuciusTTS

        if not torch.cuda.is_available():
            _state["reason"] = (
                f"CUDA-enabled PyTorch is required, but found {torch.__version__} without CUDA. "
                "Install the PyTorch cu126 build for this GPU."
            )
            print(f"  [confucius] {_state['reason']}", flush=True)
            return None
        device = "cuda"
        print(f"  [confucius] loading Confucius4-TTS on {device}...", flush=True)
        # Its upstream config deliberately uses relative checkpoint/tokenizer
        # paths. The unified app launches from ai-dubbing, so initialize from the
        # Confucius checkout rather than resolving those paths against the app.
        previous_cwd = os.getcwd()
        try:
            os.chdir(REPO)
            _state["model"] = ConfuciusTTS(config_path=CONFIG, device=device)
        finally:
            os.chdir(previous_cwd)
        _state["reason"] = ""
        print("  [confucius] ready.", flush=True)
    except Exception as error:
        _state["reason"] = str(error)[:180]
        print(f"  [confucius] unavailable ({_state['reason']}); using normal voice.", flush=True)
    return _state["model"]


def synthesize(jobs, reference_wav: str, lang: str, spk_labels=None, spk_refs=None, segment_refs=None):
    """Synthesize jobs with per-line delivery references when they are available.

    The model's prompt conditions both its speaker/style encoder and its semantic
    reference features.  A source clip for the current subtitle therefore keeps
    emotion and cadence more faithfully than using one full-video voice prompt.
    """
    model = _load()
    if model is None or not reference_wav or not os.path.isfile(reference_wav):
        return [None] * len(jobs)
    language = LANGS.get(lang)
    if not language:
        return [None] * len(jobs)
    import torchaudio

    spk_refs = spk_refs or {}
    segment_refs = segment_refs or {}
    paths = []
    for index, (text, base) in enumerate(jobs, 1):
        path = None
        text = (text or "").strip()
        if text:
            fallback_prompt = reference_wav
            if spk_labels is not None:
                candidate = spk_refs.get(int(spk_labels[index - 1]))
                if candidate and os.path.isfile(candidate):
                    fallback_prompt = candidate
            delivery_ref = segment_refs.get(index - 1)
            prompt = delivery_ref if delivery_ref and os.path.isfile(delivery_ref) else fallback_prompt
            try:
                path = base + ".wav"
                audio = model.generate(text=text, lang=language, prompt_wav=prompt, verbose=False)
                torchaudio.save(path, audio.cpu(), model.sample_rate)
                if not (os.path.isfile(path) and os.path.getsize(path) > 0):
                    path = None
            except Exception as error:
                if prompt != fallback_prompt:
                    try:
                        audio = model.generate(text=text, lang=language, prompt_wav=fallback_prompt, verbose=False)
                        torchaudio.save(path, audio.cpu(), model.sample_rate)
                        if os.path.isfile(path) and os.path.getsize(path) > 0:
                            print(f"  [confucius] segment {index} used speaker fallback reference.", flush=True)
                        else:
                            path = None
                    except Exception as fallback_error:
                        print(
                            f"  [confucius] segment {index} not voiced ({str(fallback_error)[:70]}); skipped.",
                            flush=True,
                        )
                        path = None
                else:
                    print(f"  [confucius] segment {index} not voiced ({str(error)[:70]}); skipped.", flush=True)
                    path = None
        paths.append(path)
        print(f"__PCT__ {12 + int(73 * index / max(1, len(jobs)))}", flush=True)
    return paths
