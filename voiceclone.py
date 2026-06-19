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
HERE = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(HERE, "_audio_work")


def _prepare_runtime(ffmpeg=None):
    """Give OpenVoice/numba/whisper writable local paths and bundled ffmpeg."""
    tmp = os.path.join(WORK_DIR, "tmp")
    numba_cache = os.path.join(WORK_DIR, "numba_cache")
    try:
        os.makedirs(tmp, exist_ok=True)
        os.makedirs(numba_cache, exist_ok=True)
    except Exception:
        pass
    os.environ.setdefault("TMP", tmp)
    os.environ.setdefault("TEMP", tmp)
    os.environ.setdefault("NUMBA_CACHE_DIR", numba_cache)

    candidates = []
    if ffmpeg:
        candidates.append(os.path.dirname(os.path.abspath(ffmpeg)))
    candidates.append(os.path.join(HERE, "ffmpeg-8.1.1-essentials_build", "bin"))
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for folder in candidates:
        if folder and os.path.isdir(folder) and folder not in path_parts:
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
            path_parts.insert(0, folder)

_state = {"tried": False, "converter": None, "device": None}
_target_se: dict = {}   # reference wav path -> original-speaker embedding (per video)
_source_se: dict = {}   # language code      -> base edge-tts voice embedding (per language)


def available() -> bool:
    """True if OpenVoice + torch + the converter checkpoint can actually be loaded."""
    _prepare_runtime()
    return _load() is not None


def _load():
    """Lazy-load the ToneColorConverter exactly once. Returns it, or None if unavailable."""
    _prepare_runtime()
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


def _source(seg_paths, work_dir, lang, converter):
    """Embedding of the synthesized (edge-tts/Piper) voice, computed once per language.

    Built from a concatenation of the dubbed segments rather than a single one:
    OpenVoice's VAD-based extractor needs >~5s of speech (split_seconds=10), and
    individual subtitle segments are usually 2-5s, which would all fail with
    'input audio is too short'. The synth voice is identical across a language, so
    one combined reference gives a stable source embedding for every segment.
    """
    if lang in _source_se:
        return _source_se[lang]
    from pydub import AudioSegment
    combined = AudioSegment.silent(duration=0)
    for p in seg_paths:
        if p and os.path.exists(p):
            try:
                combined += AudioSegment.from_file(p)
            except Exception:
                continue
            if combined.duration_seconds >= 30:   # plenty for a stable embedding
                break
    ref = os.path.join(work_dir, f"src_ref_{lang}.wav")
    combined.export(ref, format="wav")
    _source_se[lang] = _get_se(ref, converter)
    return _source_se[lang]


def clone_segments(seg_paths, reference_wav, work_dir, lang, ffmpeg=None):
    """Convert each dubbed segment to the original speaker's voice.

    Returns a new list of paths: converted where possible, the untouched original
    otherwise. Never raises - any failure falls back to the original segment so a dub
    is always produced.
    """
    _prepare_runtime(ffmpeg)
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
    try:
        src = _source(seg_paths, work_dir, lang, converter)
    except Exception as e:
        print(f"  [clone] couldn't model the dubbed voice ({str(e)[:60]}); skipped.", flush=True)
        return seg_paths
    if src is None:
        return seg_paths

    out = []
    for idx, p in enumerate(seg_paths):
        if not p or not os.path.exists(p):
            out.append(p)
            continue
        try:
            dst = os.path.join(work_dir, f"clone_{idx:03d}.wav")
            converter.convert(audio_src_path=p, src_se=src, tgt_se=tgt,
                              output_path=dst, message="@dub")
            out.append(dst if os.path.exists(dst) else p)
        except Exception as e:
            print(f"  [clone] segment {idx} not converted ({str(e)[:50]}); kept original.", flush=True)
            out.append(p)
    return out
