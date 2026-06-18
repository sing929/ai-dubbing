#!/usr/bin/env python3
# xtts.py - OPTIONAL natural-voice synthesis add-on for dub.py (ENGLISH only).
#
# Replaces the default Piper/edge-tts English voice with Coqui XTTS v2 - a local,
# free, far more expressive neural voice that can also CLONE the original speaker
# (so a male source is dubbed in a male voice and a female in a female one, instead
# of the fixed female Piper voice). When a reference voice is available - the Demucs
# "vocals.wav" dub.py already produces when clone / keep-music / gender mode is on -
# XTTS clones it; otherwise it falls back to a natural built-in studio speaker.
#
# This is an ADD-ON, mirroring voiceclone.py: if coqui-tts / torch / the model are
# missing, available() returns False and dub.py silently falls back to Piper, so the
# free pipeline keeps working on any machine. XTTS only handles English - Malay,
# Indonesian and Vietnamese are not supported by the model and stay on edge-tts.
#
# Install (into the Accio Python):  python -m pip install coqui-tts "transformers<5"
#   (coqui-tts 0.27 imports isin_mps_friendly, which transformers 5.x removed, so
#    transformers must stay on the 4.x line.) A CUDA GPU is strongly recommended;
#    on CPU synthesis works but is slow.

import os
import re

# Auto-agree to the Coqui model licence so the first-run model download doesn't block
# on an interactive stdin prompt (we run head-less from the GUI / .bat).
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# Force transformers (pulled in by coqui-tts) onto the torch backend only. A broken
# TensorFlow 2.15 in this machine's user site-packages can't load under NumPy 2.x, and
# transformers auto-imports any TF it detects - crashing the XTTS import. We never use
# TF here, so disable its detection. Must be set before transformers is first imported.
os.environ.setdefault("USE_TF", "0")

MODEL = os.environ.get("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
# Built-in studio speaker used when no original-voice reference is available.
DEFAULT_SPEAKER = os.environ.get("XTTS_SPEAKER", "Claribel Dervla")
# Distinct male / female built-in voices for gender mode. Picked for a big, obvious
# pitch gap (measured F0 ~90 Hz vs ~240 Hz) so the two genders sound clearly apart.
MALE_SPEAKER = os.environ.get("XTTS_MALE_SPEAKER", "Damien Black")
FEMALE_SPEAKER = os.environ.get("XTTS_FEMALE_SPEAKER", "Daisy Studious")

# The 17 languages XTTS v2 can speak, mapping the app's language code -> the code
# XTTS expects at inference (identical except Chinese: app "zh-CN" -> XTTS "zh-cn").
# dub.py gates on `lang in XTTS_LANGS`; anything not here stays on edge-tts/Piper.
XTTS_LANGS = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt",
    "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
    "zh-CN": "zh-cn", "hu": "hu", "ko": "ko", "ja": "ja", "hi": "hi",
}

# Distinct built-in studio speakers handed to detected speakers in multi-speaker mode
# when we're NOT cloning each one. Female/male interleaved so neighbours sound apart.
SPEAKER_POOL = ["Daisy Studious", "Damien Black", "Gracie Wise", "Andrew Chipper",
                "Tanja Adelina", "Viktor Eka", "Alison Dietlinde", "Craig Gutsy"]

_state = {"tried": False, "model": None, "sr": 24000}
_cond_cache: dict = {}   # reference wav path (or "<default>") -> (gpt_cond_latent, speaker_embedding)


def available() -> bool:
    """True if coqui-tts + torch + the XTTS model can actually be loaded."""
    if os.environ.get("XTTS_DISABLE"):
        return False
    return _load() is not None


def _load():
    """Lazy-load the XTTS model exactly once. Returns the Xtts model, or None."""
    if _state["tried"]:
        return _state["model"]
    _state["tried"] = True
    try:
        import torch
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("  [xtts] no CUDA GPU detected - English synthesis will be SLOW.", flush=True)
        print("  [xtts] loading XTTS v2 (first run downloads ~1.8GB, one-time)...", flush=True)
        api = TTS(MODEL).to(device)
        _state["model"] = api.synthesizer.tts_model
        _state["sr"] = getattr(api.synthesizer, "output_sample_rate", None) or 24000
        print(f"  [xtts] ready on {device}.", flush=True)
        return _state["model"]
    except Exception as e:
        print(f"  [xtts] unavailable ({str(e)[:90]}); using normal voice.", flush=True)
        return None


def _conditioning(model, reference_wav=None, speaker=None):
    """Return (gpt_cond_latent, speaker_embedding) for a reference wav (clone) or a
    named built-in speaker (the default when neither is given). Cached per reference /
    speaker so each voice is computed once, not per line."""
    key = ("ref:" + os.path.abspath(reference_wav)) if reference_wav \
        else ("spk:" + (speaker or DEFAULT_SPEAKER))
    if key in _cond_cache:
        return _cond_cache[key]
    if reference_wav and os.path.exists(reference_wav):
        gpt, spk = model.get_conditioning_latents(
            audio_path=[reference_wav], gpt_cond_len=30, max_ref_length=60)
    else:
        sm = model.speaker_manager
        name = speaker or DEFAULT_SPEAKER
        if name not in sm.speakers:
            name = DEFAULT_SPEAKER if DEFAULT_SPEAKER in sm.speakers else next(iter(sm.speakers))
        d = sm.speakers[name]
        gpt, spk = d["gpt_cond_latent"], d["speaker_embedding"]
    _cond_cache[key] = (gpt, spk)
    return gpt, spk


def synthesize(jobs, reference_wav=None, genders=None, lang="en",
               spk_labels=None, spk_refs=None):
    """Voice each segment with XTTS in language `lang` (an XTTS_LANGS key).
    jobs: list of (text, base_path_no_ext).
    reference_wav: when set, clone this voice for every line (the original speaker).
    genders: optional list of "M"/"F" per job - when given AND not cloning, each line
    is voiced with a distinct male / female built-in speaker.
    spk_labels: optional per-job speaker id (multi-speaker mode). Each speaker is voiced
    with their OWN cloned voice when spk_refs has a reference for them, else a distinct
    built-in studio speaker from SPEAKER_POOL. Takes priority over genders.
    spk_refs: {speaker_id: reference_wav} for per-speaker cloning.
    Returns a per-job .wav path, or None when a segment has no speakable text or
    synthesis fails (caller skips None). Never raises - a hard failure is caught by
    available() upstream, which falls back to Piper for the whole language."""
    n = len(jobs)
    model = _load()
    if model is None:
        return [None] * n
    try:
        import numpy as np
        import soundfile as sf
    except Exception as e:
        print(f"  [xtts] audio libs missing ({str(e)[:50]}); using normal voice.", flush=True)
        return [None] * n
    labels = spk_labels if (spk_labels and len(spk_labels) == n) else None
    spk_refs = spk_refs or {}
    # Per-line male/female voices only apply when we're NOT cloning one original voice
    # and NOT in multi-speaker mode (both carry their own per-line voice already).
    use_gender = reference_wav is None and bool(genders) and labels is None
    base_cond = None
    if not use_gender and labels is None:
        try:
            base_cond = _conditioning(model, reference_wav=reference_wav)
        except Exception as e:
            print(f"  [xtts] couldn't read the reference voice ({str(e)[:60]}); using built-in.", flush=True)
            try:
                base_cond = _conditioning(model, speaker=DEFAULT_SPEAKER)
            except Exception:
                return [None] * n

    def _cond_for(i):
        """Conditioning (gpt, speaker-embedding) for job index i, by mode priority."""
        if labels is not None:
            lab = labels[i]
            if lab in spk_refs and os.path.exists(spk_refs[lab]):
                return _conditioning(model, reference_wav=spk_refs[lab])
            return _conditioning(model, speaker=SPEAKER_POOL[lab % len(SPEAKER_POOL)])
        if use_gender:
            g = genders[i] if i < len(genders) else "F"
            return _conditioning(model, speaker=(MALE_SPEAKER if g == "M" else FEMALE_SPEAKER))
        return base_cond

    sr = _state["sr"]
    xlang = XTTS_LANGS.get(lang, "en")   # app code -> XTTS inference code
    out: list[str | None] = []
    for idx, (text, base) in enumerate(jobs, 1):
        text = (text or "").strip()
        path: str | None = None
        # Skip segments with nothing speakable (music marks, lone punctuation, symbols).
        if text and re.search(r"[^\W_]", text, re.UNICODE):
            path = base + ".wav"
            if os.path.exists(path) and os.path.getsize(path) > 0:
                out.append(path)
                print(f"__PCT__ {12 + int(73 * idx / n)}", flush=True)
                continue
            try:
                gpt, spk = _cond_for(idx - 1)
                res = model.inference(text, xlang, gpt, spk,
                                      temperature=0.7, enable_text_splitting=True)
                wav = np.asarray(res["wav"], dtype="float32")
                sf.write(path, wav, sr)
                if not (os.path.exists(path) and os.path.getsize(path) > 0):
                    path = None
            except Exception as e:
                print(f"  [xtts] segment {idx} not voiced ({str(e)[:50]}); skipped.", flush=True)
                path = None
        out.append(path)
        print(f"__PCT__ {12 + int(73 * idx / n)}", flush=True)
    return out
