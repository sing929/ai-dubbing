#!/usr/bin/env python3
# voiceclone.py - OPTIONAL per-video voice cloning add-on for dub.py.
#
# Re-colours the dubbed (edge-tts) speech so it sounds like the ORIGINAL speaker in the
# source video, using OpenVoice v2's zero-shot tone-colour converter (no training).
# The reference voice is the Demucs-isolated "vocals.wav" that dub.py already produces.
#
# This is an ADD-ON. If torch / OpenVoice / the checkpoints are missing, every function
# degrades to a no-op (returns the input untouched) so the free pipeline keeps working
# on any machine. A CUDA GPU is strongly recommended - on CPU conversion is very slow.
#
# Setup (on the GPU PC): see SETUP_VOICECLONE.md.

import os

# Where the OpenVoice v2 "converter" checkpoint lives. Override with env OPENVOICE_CKPT.
CKPT_DIR = os.environ.get(
    "OPENVOICE_CKPT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_v2", "converter"),
)

_state = {"tried": False, "converter": None, "device": None}
_target_se: dict = {}   # reference wav path -> original-speaker embedding (per video)
_source_se: dict = {}   # language code      -> base edge-tts voice embedding (per language)


def available() -> bool:
    """True if OpenVoice + torch + the converter checkpoint can actually be loaded."""
    return _load() is not None


def _load():
    """Lazy-load the ToneColorConverter exactly once. Returns it, or None if unavailable."""
    if _state["tried"]:
        return _state["converter"]
    _state["tried"] = True
    try:
        import torch
        from openvoice.api import ToneColorConverter
        cfg = os.path.join(CKPT_DIR, "config.json")
        ckpt = os.path.join(CKPT_DIR, "checkpoint.pth")
        if not (os.path.exists(cfg) and os.path.exists(ckpt)):
            print(f"  [clone] checkpoint not found in {CKPT_DIR}; see SETUP_VOICECLONE.md", flush=True)
            return None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("  [clone] no CUDA GPU detected - conversion will be SLOW.", flush=True)
        conv = ToneColorConverter(cfg, device=device)
        conv.load_ckpt(ckpt)
        _state["converter"] = conv
        _state["device"] = device
        print(f"  [clone] OpenVoice ready on {device}.", flush=True)
        return conv
    except Exception as e:
        print(f"  [clone] OpenVoice unavailable ({str(e)[:90]}); using normal voice.", flush=True)
        return None


def _get_se(wav_path, converter):
    """Speaker embedding for one audio file (uses OpenVoice's VAD-based extractor)."""
    from openvoice import se_extractor
    se, _ = se_extractor.get_se(wav_path, converter, vad=True)
    return se


def _target(reference_wav, converter):
    key = os.path.abspath(reference_wav)
    if key not in _target_se:
        _target_se[key] = _get_se(reference_wav, converter)
    return _target_se[key]


def clone_segments(seg_paths, reference_wav, work_dir, lang, ffmpeg=None):
    """Convert each dubbed segment to the original speaker's voice.

    Returns a new list of paths: converted where possible, the untouched original
    otherwise. Never raises - any failure falls back to the original segment so a dub
    is always produced.
    """
    converter = _load()
    if converter is None or not reference_wav or not os.path.exists(reference_wav):
        return seg_paths
    try:
        tgt = _target(reference_wav, converter)
    except Exception as e:
        print(f"  [clone] couldn't read the original voice ({str(e)[:60]}); skipped.", flush=True)
        return seg_paths
    if tgt is None:
        return seg_paths

    out = []
    for idx, p in enumerate(seg_paths):
        if not p or not os.path.exists(p):
            out.append(p)
            continue
        try:
            # The edge-tts base voice is identical across a language, so compute the
            # source embedding once from the first real segment and reuse it.
            if lang not in _source_se:
                _source_se[lang] = _get_se(p, converter)
            dst = os.path.join(work_dir, f"clone_{idx:03d}.wav")
            converter.convert(audio_src_path=p, src_se=_source_se[lang], tgt_se=tgt,
                              output_path=dst, message="@dub")
            out.append(dst if os.path.exists(dst) else p)
        except Exception as e:
            print(f"  [clone] segment {idx} not converted ({str(e)[:50]}); kept original.", flush=True)
            out.append(p)
    return out
