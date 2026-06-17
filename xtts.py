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

MODEL = os.environ.get("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
# Built-in studio speaker used when no original-voice reference is available.
DEFAULT_SPEAKER = os.environ.get("XTTS_SPEAKER", "Claribel Dervla")
# Distinct male / female built-in voices for gender mode. Picked for a big, obvious
# pitch gap (measured F0 ~90 Hz vs ~240 Hz) so the two genders sound clearly apart.
MALE_SPEAKER = os.environ.get("XTTS_MALE_SPEAKER", "Damien Black")
FEMALE_SPEAKER = os.environ.get("XTTS_FEMALE_SPEAKER", "Daisy Studious")

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


def synthesize(jobs, reference_wav=None, genders=None):
    """Voice each English segment with XTTS. jobs: list of (text, base_path_no_ext).
    reference_wav: when set, clone this voice for every line (the original speaker).
    genders: optional list of "M"/"F" per job - when given AND not cloning, each line
    is voiced with a distinct male / female built-in speaker so the original speakers'
    genders stay clearly apart. Returns a per-job .wav path, or None when a segment has
    no speakable text or synthesis fails (caller skips None, exactly like tts_all).
    Never raises - a hard failure is caught by available() upstream, which falls back
    to Piper for the whole language."""
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
    # Per-line male/female voices only apply when we're NOT cloning one original voice
    # (a clone already carries the original speaker's own gender).
    use_gender = reference_wav is None and bool(genders)
    base_cond = None
    if not use_gender:
        try:
            base_cond = _conditioning(model, reference_wav=reference_wav)
        except Exception as e:
            print(f"  [xtts] couldn't read the reference voice ({str(e)[:60]}); using built-in.", flush=True)
            try:
                base_cond = _conditioning(model, speaker=DEFAULT_SPEAKER)
            except Exception:
                return [None] * n

    sr = _state["sr"]
    out: list[str | None] = []
    for idx, (text, base) in enumerate(jobs, 1):
        text = (text or "").strip()
        path: str | None = None
        # Skip segments with nothing speakable (music marks, lone punctuation, symbols).
        if text and re.search(r"[^\W_]", text, re.UNICODE):
            path = base + ".wav"
            try:
                if use_gender:
                    g = genders[idx - 1] if idx - 1 < len(genders) else "F"
                    gpt, spk = _conditioning(
                        model, speaker=(MALE_SPEAKER if g == "M" else FEMALE_SPEAKER))
                else:
                    gpt, spk = base_cond
                res = model.inference(text, "en", gpt, spk,
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
