#!/usr/bin/env python3
# moss_tts.py - OPTIONAL MOSS-TTS-Nano ONNX voice backend for dub.py.
#
# This mirrors xtts.py / voiceclone.py: if the MOSS checkout, ONNX runtime, or
# model assets are missing, available() returns False and dub.py keeps using the
# existing voices. Enable with MOSS_TTS_ENABLE=1. Override the checkout with
# MOSS_TTS_REPO=E:\dub\MOSS-TTS-Nano.

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.abspath(os.path.join(HERE, "..", "MOSS-TTS-Nano"))
REPO = os.environ.get("MOSS_TTS_REPO", DEFAULT_REPO)
MODEL_DIR = os.environ.get("MOSS_TTS_MODEL_DIR", os.path.join(REPO, "models"))
EXECUTION_PROVIDER = os.environ.get("MOSS_TTS_PROVIDER", "cpu").strip().lower() or "cpu"
CPU_THREADS = int(os.environ.get("MOSS_TTS_CPU_THREADS", "4") or "4")

MOSS_LANGS = {
    "zh-CN", "en", "de", "es", "fr", "ja", "it", "hu", "ko", "ru",
    "ar", "pl", "pt", "cs", "tr",
}

DEFAULT_VOICE_BY_LANG = {
    "zh-CN": "Junhao",
    "en": "Ava",
    "ja": "Sakura",
}

MALE_VOICES = ["Adam", "Nathan", "Junhao", "Zhiming", "Weiguo"]
FEMALE_VOICES = ["Ava", "Bella", "Xiaoyu", "Yuewen", "Lingyu", "Sakura", "Yui", "Aoi", "Hina", "Mei"]
SPEAKER_POOL = ["Ava", "Adam", "Bella", "Nathan", "Xiaoyu", "Junhao", "Yuewen", "Zhiming", "Sakura", "Yui"]

_state = {"tried": False, "runtime": None}


def enabled() -> bool:
    return os.environ.get("MOSS_TTS_ENABLE", "").strip().lower() in {"1", "true", "yes", "on"}


def available() -> bool:
    if not enabled() or os.environ.get("MOSS_TTS_DISABLE"):
        return False
    return _load() is not None


def _load():
    if _state["tried"]:
        return _state["runtime"]
    _state["tried"] = True
    if not os.path.exists(os.path.join(REPO, "infer_onnx.py")):
        print(f"  [moss] checkout not found at {REPO}; using normal voice.", flush=True)
        return None
    try:
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from onnx_tts_runtime import OnnxTtsRuntime

        print(f"  [moss] loading MOSS-TTS-Nano ONNX ({EXECUTION_PROVIDER})...", flush=True)
        runtime = OnnxTtsRuntime(
            model_dir=MODEL_DIR,
            thread_count=CPU_THREADS,
            execution_provider=EXECUTION_PROVIDER,
        )
        _state["runtime"] = runtime
        print("  [moss] ready.", flush=True)
        return runtime
    except Exception as e:
        print(f"  [moss] unavailable ({str(e)[:100]}); using normal voice.", flush=True)
        return None


def _voice_for(lang: str, idx: int, labels=None, genders=None) -> str:
    if labels is not None:
        lab = int(labels[idx])
        return SPEAKER_POOL[lab % len(SPEAKER_POOL)]
    if genders is not None and idx < len(genders):
        if genders[idx] == "M":
            return MALE_VOICES[idx % len(MALE_VOICES)]
        return FEMALE_VOICES[idx % len(FEMALE_VOICES)]
    return DEFAULT_VOICE_BY_LANG.get(lang, "Ava")


def synthesize(jobs, reference_wav=None, genders=None, lang="en",
               spk_labels=None, spk_refs=None):
    """Voice each segment with MOSS-TTS-Nano ONNX.

    jobs: list of (text, base_path_no_ext). Returns per-job wav path or None.
    reference_wav clones one voice for all lines. spk_refs can provide one clone
    reference per speaker id. Built-in voice presets are used otherwise.
    """
    n = len(jobs)
    runtime = _load()
    if runtime is None:
        return [None] * n

    labels = spk_labels if (spk_labels and len(spk_labels) == n) else None
    spk_refs = spk_refs or {}
    out = []
    for idx, (text, base) in enumerate(jobs, 1):
        text = (text or "").strip()
        path = None
        if text and re.search(r"[^\W_]", text, re.UNICODE):
            path = base + ".wav"
            if os.path.exists(path) and os.path.getsize(path) > 0:
                out.append(path)
                print(f"__PCT__ {12 + int(73 * idx / n)}", flush=True)
                continue
            try:
                prompt = None
                if labels is not None and int(labels[idx - 1]) in spk_refs:
                    cand = spk_refs[int(labels[idx - 1])]
                    prompt = cand if cand and os.path.exists(cand) else None
                elif reference_wav and os.path.exists(reference_wav):
                    prompt = reference_wav
                runtime.synthesize(
                    text=text,
                    voice=_voice_for(lang, idx - 1, labels, genders),
                    prompt_audio_path=prompt,
                    output_audio_path=path,
                    enable_wetext=False,
                    enable_normalize_tts_text=True,
                )
                if not (os.path.exists(path) and os.path.getsize(path) > 0):
                    path = None
            except Exception as e:
                print(f"  [moss] segment {idx} not voiced ({str(e)[:70]}); skipped.", flush=True)
                path = None
        out.append(path)
        print(f"__PCT__ {12 + int(73 * idx / n)}", flush=True)
    return out
