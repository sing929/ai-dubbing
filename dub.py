#!/usr/bin/env python3
# dub.py - transcribe a video and produce DUBBED copies in target languages.
# 100% free + local: faster-whisper (speech->text) -> deep-translator (Google MT)
# -> edge-tts (Microsoft neural voices). The original voice is removed by reusing
# the Demucs "no_vocals" music bed (same one MusicRemove makes); the new voice is
# mixed on top of the music and muxed back onto the video. A translated .srt is
# saved next to each dubbed file.
#
# Usage:
#   python dub.py <video> <out_dir> <ffmpeg.exe> <work_dir> <langs> [model]
#   langs  = comma list from: vi (Vietnamese), id (Indonesian), es (Spanish/LatAm)
#   model  = faster-whisper size: tiny|base|small|medium  (default: medium)

import sys
import os
import re
import time
import json
import hashlib
import asyncio
import subprocess
import importlib.util
from multimodal_aligner import (
    MultimodalAligner,
    apply_multimodal_context,
    enforce_alignment_constraints,
)

# Quieter first-run model download (these are harmless Windows/HF notices).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Force the torch backend for transformers (pulled in by coqui-tts / XTTS). This
# machine has a broken TensorFlow 2.15 (can't load under NumPy 2.x) in user
# site-packages; transformers auto-imports any TF it detects, which crashes the
# XTTS English voice and gets logged as an error. faster-whisper/demucs import
# transformers before xtts.py runs, so this MUST be set here at the top - before
# any import below - not in xtts.py. We never use TF here.
os.environ.setdefault("USE_TF", "0")


def ensure_deps() -> None:
    """Install the free pip tools on first run so the .bat stays no-coding."""
    need = {
        "faster_whisper": "faster-whisper",
        "deep_translator": "deep-translator",
        "edge_tts": "edge-tts",
        "pydub": "pydub",
        "piper": "piper-tts",
    }
    missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("  installing:", ", ".join(missing))
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing])


ensure_deps()


def _preinit_torch_cudnn() -> None:
    """Load torch's cuDNN before CTranslate2 (faster-whisper) loads its own.

    On Windows with the optional voice-clone add-on installed, importing
    faster-whisper first makes CTranslate2 load an incompatible cuDNN; torch's
    later load then dies with "Could not load symbol cudnnGetLibConfig" (exit
    127), killing the whole dub. Forcing a tiny CUDA cuDNN op here pins torch's
    cuDNN first and avoids the conflict. No-op without torch/CUDA.
    """
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.randn(1, 1, 4, 4, device="cuda")
            torch.nn.functional.conv2d(x, torch.randn(1, 1, 3, 3, device="cuda"))
            torch.cuda.synchronize()
    except Exception:
        pass


_preinit_torch_cudnn()

from faster_whisper import WhisperModel          # noqa: E402
from deep_translator import GoogleTranslator, MyMemoryTranslator  # noqa: E402
import edge_tts                                    # noqa: E402
import warnings                                     # noqa: E402
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")
from pydub import AudioSegment                     # noqa: E402
from feedback_storage import FeedbackStorage       # noqa: E402
try:
    import numpy as np                              # noqa: E402  (ships with faster-whisper)
except Exception:
    np = None

# Natural female neural voices. Change a value to a *Neural male voice if you prefer.
# This dict is ALSO the master list of languages the tool accepts (run_one filters
# the requested langs to keys present here). Languages marked [XTTS] also get the
# local XTTS v2 natural voice + cloning (see xtts.XTTS_LANGS); the rest use edge-tts.
VOICES = {
    # edge-tts only (XTTS v2 doesn't model these)
    "vi": "vi-VN-HoaiMyNeural",   # Vietnamese (female)
    "id": "id-ID-GadisNeural",    # Indonesian (female)
    "ms": "ms-MY-YasminNeural",   # Malay (female)
    # [XTTS] the 17 languages XTTS v2 supports
    "en": "en-US-AriaNeural",     # English (US, female)
    "es": "es-MX-DaliaNeural",    # Spanish - Latin America / Mexico (female)
    "fr": "fr-FR-DeniseNeural",   # French (female)
    "de": "de-DE-KatjaNeural",    # German (female)
    "it": "it-IT-ElsaNeural",     # Italian (female)
    "pt": "pt-BR-FranciscaNeural",  # Portuguese (Brazil, female)
    "pl": "pl-PL-ZofiaNeural",    # Polish (female)
    "tr": "tr-TR-EmelNeural",     # Turkish (female)
    "ru": "ru-RU-SvetlanaNeural",  # Russian (female)
    "nl": "nl-NL-ColetteNeural",  # Dutch (female)
    "cs": "cs-CZ-VlastaNeural",   # Czech (female)
    "ar": "ar-EG-SalmaNeural",    # Arabic (Egypt, female)
    "zh-CN": "zh-CN-XiaoxiaoNeural",  # Chinese, Simplified (female)
    "hu": "hu-HU-NoemiNeural",    # Hungarian (female)
    "ko": "ko-KR-SunHiNeural",    # Korean (female)
    "ja": "ja-JP-NanamiNeural",   # Japanese (female)
    "hi": "hi-IN-SwaraNeural",    # Hindi (female)
}

# Piper local voices (free, no throttling). Indonesian has no Piper voice, so it
# stays on edge-tts above. Swap any value below for a different Piper voice id
# (see https://github.com/rhasspy/piper/blob/master/VOICES.md).
PIPER_VOICES = {
    "vi": "vi_VN-vais1000-medium",
    "es": "es_MX-claude-high",
    "en": "en_US-amy-medium",
}

_PIPER_CACHE: dict = {}

# Nominal speaking pitch (Hz) of the AI voices, used only by tone="original" to
# nudge the dub toward the original speaker's pitch. All default voices are female
# (~190 Hz). If you swap in a male voice above, drop this toward ~110.
VOICE_REF_F0 = 190.0
VOICE_REF_F0_MALE = 115.0   # male AI voices sit ~115 Hz; used per-segment in gender mode

# Gender mode: auto-pick a male or female neural voice per line from the original
# speaker's pitch (F0). edge-tts has a clean male/female pair for every language we
# support, so gender mode always voices with edge-tts (online), not Piper. F0 below
# the split reads as male. Heuristic only - deep women / high men / noisy lines can
# misclassify, and it can't tell two same-gender speakers apart.
GENDER_F0_SPLIT = 155.0
GENDER_VOICES = {
    "vi": {"M": "vi-VN-NamMinhNeural", "F": "vi-VN-HoaiMyNeural"},
    "id": {"M": "id-ID-ArdiNeural", "F": "id-ID-GadisNeural"},
    "ms": {"M": "ms-MY-OsmanNeural", "F": "ms-MY-YasminNeural"},
    "en": {"M": "en-US-GuyNeural", "F": "en-US-AriaNeural"},
    "es": {"M": "es-MX-JorgeNeural", "F": "es-MX-DaliaNeural"},
    "fr": {"M": "fr-FR-HenriNeural", "F": "fr-FR-DeniseNeural"},
    "de": {"M": "de-DE-ConradNeural", "F": "de-DE-KatjaNeural"},
    "it": {"M": "it-IT-DiegoNeural", "F": "it-IT-ElsaNeural"},
    "pt": {"M": "pt-BR-AntonioNeural", "F": "pt-BR-FranciscaNeural"},
    "pl": {"M": "pl-PL-MarekNeural", "F": "pl-PL-ZofiaNeural"},
    "tr": {"M": "tr-TR-AhmetNeural", "F": "tr-TR-EmelNeural"},
    "ru": {"M": "ru-RU-DmitryNeural", "F": "ru-RU-SvetlanaNeural"},
    "nl": {"M": "nl-NL-MaartenNeural", "F": "nl-NL-ColetteNeural"},
    "cs": {"M": "cs-CZ-AntoninNeural", "F": "cs-CZ-VlastaNeural"},
    "ar": {"M": "ar-EG-ShakirNeural", "F": "ar-EG-SalmaNeural"},
    "zh-CN": {"M": "zh-CN-YunxiNeural", "F": "zh-CN-XiaoxiaoNeural"},
    "hu": {"M": "hu-HU-TamasNeural", "F": "hu-HU-NoemiNeural"},
    "ko": {"M": "ko-KR-InJoonNeural", "F": "ko-KR-SunHiNeural"},
    "ja": {"M": "ja-JP-KeitaNeural", "F": "ja-JP-NanamiNeural"},
    "hi": {"M": "hi-IN-MadhurNeural", "F": "hi-IN-SwaraNeural"},
}

# Multi-speaker mode: a pool of DISTINCT edge-tts voices per language. When the tool
# detects N speakers it hands each their own voice by cycling this pool (female/male
# interleaved so neighbours sound apart). Languages not listed fall back to their
# female+male GENDER_VOICES pair via _speaker_voices(); XTTS languages instead clone
# each speaker or use XTTS's own built-in speaker pool (see xtts.SPEAKER_POOL).
SPEAKER_VOICES = {
    "en": ["en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
           "en-US-ChristopherNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural"],
    "es": ["es-MX-DaliaNeural", "es-MX-JorgeNeural", "es-ES-ElviraNeural",
           "es-ES-AlvaroNeural", "es-CO-SalomeNeural", "es-AR-TomasNeural"],
    "vi": ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"],
    "id": ["id-ID-GadisNeural", "id-ID-ArdiNeural"],
    "ms": ["ms-MY-YasminNeural", "ms-MY-OsmanNeural"],
    "fr": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural", "fr-FR-EloiseNeural", "fr-CA-JeanNeural"],
    "de": ["de-DE-KatjaNeural", "de-DE-ConradNeural", "de-DE-AmalaNeural", "de-DE-KillianNeural"],
    "it": ["it-IT-ElsaNeural", "it-IT-DiegoNeural", "it-IT-IsabellaNeural", "it-IT-GiuseppeNeural"],
    "pt": ["pt-BR-FranciscaNeural", "pt-BR-AntonioNeural", "pt-BR-BrendaNeural", "pt-BR-FabioNeural"],
    "ru": ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
    "zh-CN": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunjianNeural"],
    "ja": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural", "ja-JP-AoiNeural", "ja-JP-DaichiNeural"],
    "ko": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural", "ko-KR-JiMinNeural", "ko-KR-HyunsuNeural"],
}


def _speaker_voices(lang: str) -> list[str]:
    """Ordered list of distinct edge-tts voices to hand out to detected speakers in
    `lang`. Falls back to the female+male gender pair, then the single default voice."""
    pool = SPEAKER_VOICES.get(lang)
    if pool:
        return pool
    gv = GENDER_VOICES.get(lang, {})
    pair = [v for v in (gv.get("F"), gv.get("M")) if v]
    return pair or [VOICES.get(lang, "en-US-AriaNeural")]


def _speaker_gender_voices(lang: str, spk_labels: list[int],
                           genders: list[str]) -> list[str]:
    """Voice per line when BOTH multi-speaker and gender mode are on: every distinct
    speaker is voiced in a voice of THEIR majority gender (a male speaker is never
    handed a female voice), and same-gender speakers cycle that gender's voice list
    so they still sound apart. Without this, the speaker round-robin ignored gender
    entirely and the male/female choice appeared to do nothing."""
    from collections import Counter
    fallback = VOICES.get(lang, "en-US-AriaNeural")
    pool = SPEAKER_VOICES.get(lang, [])
    # SPEAKER_VOICES is kept female/male interleaved. Split it so two male
    # characters, or two female characters, do not collapse into the same voice.
    if pool:
        pools = {"F": pool[0::2] or [fallback], "M": pool[1::2] or [fallback]}
    else:
        gv = GENDER_VOICES.get(lang, {})
        pools = {"M": [v for v in (gv.get("M"),) if v] or [fallback],
                 "F": [v for v in (gv.get("F"),) if v] or [fallback]}
    # A speaker's lines should all share one gender: take the majority vote.
    votes: dict[int, Counter] = {}
    for lab, g in zip(spk_labels, genders):
        votes.setdefault(lab, Counter())[g] += 1
    spk_gender = {lab: c.most_common(1)[0][0] for lab, c in votes.items()}
    order: dict[str, dict[int, int]] = {"M": {}, "F": {}}

    def _voice(lab: int) -> str:
        g = spk_gender[lab]
        idx = order[g].setdefault(lab, len(order[g]))
        pool = pools[g]
        return pool[idx % len(pool)]

    return [_voice(lab) for lab in spk_labels]

# Hard ceiling on how much a long line (translations often run longer than the
# source) may be sped up to fit its slot. Beyond this it sounds chipmunky, so
# instead we let the timeline drift and recover at the next pause. 1.8x is the
# practical limit before intelligibility suffers; raise/lower to trade sync vs speed.
MAX_FIT_TEMPO = 1.45
# Let an over-long dubbed line push the next line only a little. Bigger drift keeps
# voices separated but quickly makes the dub feel unsynced with the video.
MAX_SYNC_DRIFT_MS = 350

# Speak edge-tts (cloud) lines a bit faster at synthesis time. Translations from a
# terse source language (e.g. Chinese -> Indonesian) run longer than their original
# slot; generating faster speech reduces how much we must time-stretch afterwards,
# which keeps the dub better synced. Set to "+0%" to disable.
EDGE_TTS_RATE = "+6%"


def ensure_piper_voice(lang: str) -> str:
    voice_id = PIPER_VOICES[lang]
    voices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "piper_voices")
    os.makedirs(voices_dir, exist_ok=True)
    onnx = os.path.join(voices_dir, voice_id + ".onnx")
    cfg = onnx + ".json"
    if os.path.exists(onnx) and os.path.exists(cfg):
        return onnx
    import urllib.request
    lang_region, voice_name, quality = voice_id.split("-")
    short_lang = lang_region.split("_")[0]
    base = (f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            f"{short_lang}/{lang_region}/{voice_name}/{quality}/{voice_id}")
    print(f"  downloading Piper voice for {lang} ({voice_id}, one-time)...", flush=True)
    urllib.request.urlretrieve(base + ".onnx", onnx)
    urllib.request.urlretrieve(base + ".onnx.json", cfg)
    return onnx


def _piper_voice(lang: str):
    if lang in _PIPER_CACHE:
        return _PIPER_CACHE[lang]
    model_path = ensure_piper_voice(lang)
    from piper import PiperVoice
    voice = PiperVoice.load(model_path)
    _PIPER_CACHE[lang] = voice
    return voice


def run(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def ffprobe_duration(path: str, ffmpeg: str) -> float:
    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
    if not os.path.exists(ffprobe):
        ffprobe = "ffprobe"
    r = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return len(AudioSegment.from_file(path)) / 1000.0


def srt_ts(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: str, segs: list[dict], texts: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, (s, t) in enumerate(zip(segs, texts), 1):
            f.write(f"{i}\n{srt_ts(s['start'])} --> {srt_ts(s['end'])}\n{(t or '').strip()}\n\n")


def _seg_base(seg_dir: str, idx: int, text: str, voice: str = "") -> str:
    sig = hashlib.sha1(((text or "") + "\0" + (voice or "")).encode("utf-8")).hexdigest()[:10]
    return os.path.join(seg_dir, f"{idx:03d}_{sig}")


def write_project(path: str, *, base: str, source: str, output: str, audio_only: bool,
                  src_lang: str, lang: str, segs: list[dict], texts: list[str],
                  spk_labels: list[int] | None = None,
                  seg_voices: list[str] | None = None,
                  cast_roles: list[str] | None = None,
                  cast_genders: list[str] | None = None,
                  preset: str = "", rights_mode: str = "",
                  story_bible: dict | None = None) -> None:
    """Persist a per-run project file (read by web_editor.py) holding the source/output
    pairing plus the exact per-line source, translation, speaker id and voice. This is
    what lets the editor show the RIGHT translated video and real speaker labels without
    re-translating. Best-effort: a failure here must never break the dub."""
    if src_lang in ("", "auto") and segs:
        src_lang = segs[0].get("lang", "auto")
    segments = []
    for i, s in enumerate(segs):
        segments.append({
            "i": i,
            "start": round(float(s.get("start", 0.0)), 3),
            "end": round(float(s.get("end", 0.0)), 3),
            "src": s.get("text", ""),
            "tr": texts[i] if i < len(texts) else "",
            "speaker": int(spk_labels[i]) if (spk_labels is not None and i < len(spk_labels)) else 0,
            "role": cast_roles[i] if (cast_roles is not None and i < len(cast_roles)) else None,
            "gender": cast_genders[i] if (cast_genders is not None and i < len(cast_genders)) else None,
            "voice": seg_voices[i] if (seg_voices is not None and i < len(seg_voices)) else None,
            "source_track": s.get("source"),
            "visual_context": s.get("visual_context") or {},
            "visual_start": s.get("visual_start"),
            "visual_end": s.get("visual_end"),
            "visual_budget": s.get("visual_budget"),
            "lip_activity_confidence": s.get("lip_activity_confidence"),
        })
    data = {
        "version": 1, "base": base,
        "source": os.path.abspath(source), "output": os.path.abspath(output),
        "audio_only": audio_only, "src_lang": src_lang, "lang": lang,
        "preset": preset, "rights_mode": rights_mode,
        "story_bible": story_bible or {},
        "multimodal_summary": (story_bible or {}).get("visual_summary", ""),
        "created": int(time.time()),
        "duration": round(float(segs[-1]["end"]), 3) if segs else 0.0,
        "segments": segments,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"  [warn] could not write project file: {e}", flush=True)


def ffprobe_dims(path: str, ffmpeg: str) -> tuple[int, int]:
    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
    if not os.path.exists(ffprobe):
        ffprobe = "ffprobe"
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True,
    )
    try:
        w, h = r.stdout.strip().split("x")[:2]
        return int(w), int(h)
    except Exception:
        return 1080, 1920


def _parse_rect(cover_region: str, subs_band: int) -> tuple[list[float], bool]:
    """Return (rect[x,y,w,h] fractions, user_drawn). Falls back to a bottom band."""
    if cover_region:
        try:
            vals = [max(0.0, min(1.0, float(v))) for v in cover_region.split(",")]
            if len(vals) == 4 and vals[2] > 0.02 and vals[3] > 0.02:
                return vals, True
        except Exception:
            pass
    b = max(1, min(40, subs_band)) / 100.0
    return [0.0, 1.0 - b, 1.0, b], False


def ass_ts(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = min(99, int(round((sec - int(sec)) * 100)))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape_one_line(text: str) -> str:
    """ASS-safe subtitle text forced onto one visual line."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text.replace("\\", "/").replace("{", "(").replace("}", ")")


def _one_line_font_size(text: str, box_w: int, box_h: int,
                        frame_h: int, size_from_rect: bool) -> int:
    """Pick a font size that fits one line inside the original subtitle box."""
    max_h = max(16, int(box_h * (0.62 if size_from_rect else 0.38)))
    base = int(_clamp(max_h, 18, round(frame_h * 0.045)))
    max_w = max(80, box_w * 0.92)
    # Rough visual width: CJK/fullwidth chars are about 1em, Latin about 0.55em.
    units = sum(1.0 if ord(ch) > 255 else 0.55 for ch in text)
    if units > 0:
        base = min(base, int(max_w / units))
    return int(_clamp(base, 16, round(frame_h * 0.045)))


def write_ass(path: str, segs: list[dict], texts: list[str], frame_w: int,
              frame_h: int, rect: list[float] | None, size_from_rect: bool) -> None:
    """Write one-line burned subtitles inside the original subtitle area only."""
    if rect and len(rect) == 4:
        x, y, w, h = rect
    else:
        x, y, w, h = 0.0, 0.80, 1.0, 0.16
    x1 = max(0, round(x * frame_w))
    y1 = max(0, round(y * frame_h))
    x2 = min(frame_w, round((x + w) * frame_w))
    y2 = min(frame_h, round((y + h) * frame_h))
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    cx = x1 + box_w // 2
    cy = y1 + box_h // 2
    default_fs = int(_clamp(round(box_h * 0.38), 18, round(frame_h * 0.045)))
    outline = max(2, round(default_fs * 0.07))
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {frame_w}\n"
        f"PlayResY: {frame_h}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Arial,{default_fs},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        f"0,0,0,0,100,100,0,0,1,{outline},0,5,0,0,0,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for s, t in zip(segs, texts):
            t = (t or "").strip()
            if not t:
                continue
            t = _ass_escape_one_line(t)
            fs = _one_line_font_size(t, box_w, box_h, frame_h, size_from_rect)
            tag = f"{{\\an5\\pos({cx},{cy})\\clip({x1},{y1},{x2},{y2})\\fs{fs}}}"
            f.write(f"Dialogue: 0,{ass_ts(s['start'])},{ass_ts(s['end'])},"
                    f"Default,,0,0,0,,{tag}{t}\n")


def ensure_bed(video: str, base: str, work: str, ffmpeg: str) -> tuple[str | None, str | None]:
    """Return (music_bed, clean_vocals) paths, reusing Demucs output only when it
    was produced from THIS video. Two different videos that share a filename (the
    browser saves every download as 'download.mp4') must NOT reuse each other's
    separated voice - otherwise the dub clones the previous video's speaker. The
    cached output is therefore tagged with the source's size+mtime, the same
    fingerprint load_analysis() uses, and re-separated when that changes."""
    bed = os.path.join(work, "htdemucs", base, "no_vocals.wav")
    vocals = os.path.join(work, "htdemucs", base, "vocals.wav")
    sig_path = os.path.join(work, "htdemucs", base, ".source.json")
    cur_sig = _video_sig(video)

    def _cached_sig() -> dict | None:
        try:
            with open(sig_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    if os.path.exists(bed) and _cached_sig() == cur_sig:
        return bed, (vocals if os.path.exists(vocals) else None)

    stem = os.path.join(work, base + ".wav")
    run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-i", video, "-vn", "-ac", "2", "-ar", "44100", stem])
    print("  separating voice/music with Demucs (first run downloads ~80MB model)...")
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems", "vocals", "-o", work, stem])
    if os.path.exists(bed):
        try:
            with open(sig_path, "w", encoding="utf-8") as f:
                json.dump(cur_sig, f)
        except Exception:
            pass
        return bed, (vocals if os.path.exists(vocals) else None)
    print("  Demucs unavailable; original music will be dropped (voice-only dub).")
    return None, None


# Unicode blocks that pin a line to one language. Han (CJK) alone is ambiguous
# (shared by Chinese and Japanese), so KANA decides Japanese and HANGUL decides
# Korean; a line with only Han characters is read as Chinese.
_RE_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
_RE_KANA = re.compile(r"[぀-ゟ゠-ヿ]")          # hiragana + katakana
_RE_HAN = re.compile(r"[㐀-鿿豈-﫿]")


def _script_lang(text: str) -> str | None:
    """Best-guess language of one transcribed line from its script. Returns a
    source code (zh/ja/ko) or None when the script doesn't pin a language (e.g.
    Latin/Cyrillic, which the acoustic detector already labelled)."""
    if _RE_HANGUL.search(text):
        return "ko"
    if _RE_KANA.search(text):
        return "ja"
    if _RE_HAN.search(text):
        return "zh"
    return None


def transcribe(model, audio_path: str, total_ms: int = 0) -> tuple[list[dict], str]:
    print("__STAGE__ Transcribing audio", flush=True)
    # Douyin clips are music-heavy; without these, Whisper loops and re-emits the last
    # line as hundreds of duplicate Chinese segments (the "repeating" bug). Disabling
    # previous-text conditioning + an n-gram repeat guard stops the loop.
    #   multilingual=True            -> detect language PER chunk, so a mostly-Chinese
    #                                   clip with stray Korean/Japanese lines transcribes
    #                                   each in its own language instead of forcing one.
    #   language_detection_segments  -> sample several windows (not just the first) so the
    #                                   dominant language is right even if it opens on music.
    # VAD gate kept (it stops the music-loop bug) but deliberately permissive:
    # short/quiet in-video dialogue often sits under narration, SFX, or music and was
    # being dropped before translation. We prefer a few extra subtitle lines over
    # missing obvious dialogue; Demucs vocals (when available) keeps false positives
    # tolerable.
    segments, info = model.transcribe(
        audio_path, vad_filter=True, multilingual=True,
        vad_parameters=dict(threshold=0.22, min_silence_duration_ms=180,
                            min_speech_duration_ms=80, speech_pad_ms=320),
        language_detection_segments=8, language_detection_threshold=0.35,
        condition_on_previous_text=False, no_repeat_ngram_size=3)
    out: list[dict] = []
    seen: dict[str, int] = {}        # source code -> total characters, to rank the mix
    for s in segments:
        text = (s.text or "").strip()
        if text:
            slang = _script_lang(text) or info.language
            out.append({"start": s.start, "end": s.end, "text": text, "lang": slang})
            seen[slang] = seen.get(slang, 0) + len(text)
        if total_ms > 0:
            print(f"__PCT__ {min(99, int(s.end * 1000 / total_ms * 100))}", flush=True)
    # Dominant source = the language covering the most transcribed text (falls back to
    # Whisper's own guess when nothing scriptable was found, e.g. an all-Latin clip).
    dominant = max(seen, key=seen.get) if seen else info.language
    others = sorted((l for l in seen if l != dominant), key=seen.get, reverse=True)
    mix = f"   (also: {', '.join(others)})" if others else ""
    print(f"  source language: {dominant}{mix}   segments: {len(out)}", flush=True)
    return out, dominant


def _dominant_lang(segs: list[dict], fallback: str = "auto") -> str:
    seen: dict[str, int] = {}
    for s in segs:
        lang = s.get("lang") or fallback
        seen[lang] = seen.get(lang, 0) + len(s.get("text", ""))
    return max(seen, key=seen.get) if seen else fallback


def _overlap_ratio(a: dict, b: dict) -> float:
    lo = max(float(a.get("start", 0.0)), float(b.get("start", 0.0)))
    hi = min(float(a.get("end", 0.0)), float(b.get("end", 0.0)))
    ov = max(0.0, hi - lo)
    dur = max(0.01, min(float(a.get("end", 0.0)) - float(a.get("start", 0.0)),
                        float(b.get("end", 0.0)) - float(b.get("start", 0.0))))
    return ov / dur


def merge_transcripts(primary: list[dict], fallback: list[dict]) -> tuple[list[dict], int]:
    """Add fallback full-mix lines that do not overlap the cleaner vocals transcript.

    Demucs vocals usually improve transcription, but it can remove quiet in-scene
    dialogue along with music/SFX. A second full-mix pass can recover those missing
    lines; we only add lines that are temporally distinct so duplicates stay rare.
    """
    out = [dict(s, source=s.get("source", "vocals")) for s in primary]
    added = 0
    for s in fallback:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        if any(_overlap_ratio(s, p) >= 0.35 for p in out):
            continue
        out.append(dict(s, source="fullmix"))
        added += 1
    out.sort(key=lambda x: (float(x.get("start", 0.0)), float(x.get("end", 0.0))))
    return out, added


def transcribe_for_dub(model, video: str, vocals: str | None, total_ms: int,
                      dual_pass: bool = False) -> tuple[list[dict], str, dict]:
    """Transcribe with the best available source and optional full-mix recovery."""
    if vocals and os.path.exists(vocals):
        segs, src_lang = transcribe(model, vocals, total_ms)
        for s in segs:
            s["source"] = "vocals"
        meta = {"primary": "vocals", "dual_pass": False, "added_fullmix": 0}
        if dual_pass:
            print("__STAGE__ Transcribing full mix for missed dialogue", flush=True)
            alt, _alt_lang = transcribe(model, video, total_ms)
            segs, added = merge_transcripts(segs, alt)
            src_lang = _dominant_lang(segs, src_lang)
            meta.update({"dual_pass": True, "alt_segments": len(alt),
                         "added_fullmix": added})
            if added:
                print(f"  recovered {added} extra dialogue lines from full mix", flush=True)
        return segs, src_lang, meta
    segs, src_lang = transcribe(model, video, total_ms)
    for s in segs:
        s["source"] = "fullmix"
    return segs, src_lang, {"primary": "fullmix", "dual_pass": False, "added_fullmix": 0}


def tts_all(jobs: list[tuple[str, str]], lang: str,
            seg_voices: list[str] | None = None) -> list[str | None]:
    """Synthesize each segment to file. Piper (local, no throttling) for languages
    that have a voice; edge-tts (cloud) as fallback (currently just Indonesian).
    jobs: list of (text, base_path_no_ext). Returns: per-job output path or None.
    seg_voices: when given (gender mode), a per-segment edge-tts voice id - each
    line is voiced online with its own voice instead of the single language voice."""
    n = len(jobs)
    paths: list[str | None] = []
    if seg_voices is not None:
        return asyncio.run(_edge_tts_all(jobs, seg_voices))
    if lang in PIPER_VOICES:
        try:
            voice = _piper_voice(lang)
            import wave
            for idx, (text, base) in enumerate(jobs, 1):
                text = (text or "").strip()
                path: str | None = None
                # Skip segments with nothing speakable (music marks, lone punctuation, symbols).
                if text and re.search(r"[^\W_]", text, re.UNICODE):
                    path = base + ".wav"
                    if os.path.exists(path) and os.path.getsize(path) > 0:
                        paths.append(path)
                        print(f"__PCT__ {12 + int(73 * idx / n)}", flush=True)
                        continue
                    try:
                        # piper-tts >=1.3 returns AudioChunk objects from synthesize();
                        # we have to set the wave params from the first chunk ourselves.
                        with wave.open(path, "wb") as f:
                            params_set = False
                            for chunk in voice.synthesize(text):
                                if not params_set:
                                    f.setnchannels(getattr(chunk, "sample_channels", 1))
                                    f.setsampwidth(getattr(chunk, "sample_width", 2))
                                    f.setframerate(getattr(chunk, "sample_rate", 22050))
                                    params_set = True
                                f.writeframes(getattr(chunk, "audio_int16_bytes", b""))
                    except Exception as e:
                        print("   piper skipped 1:", str(e)[:50])
                        path = None
                paths.append(path)
                print(f"__PCT__ {12 + int(73 * idx / n)}", flush=True)
            return paths
        except Exception as e:
            print(f"  Piper unavailable ({str(e)[:60]}); falling back to cloud voice", flush=True)
    # Cloud fallback: edge-tts (Indonesian, or Piper load failure).
    return asyncio.run(_edge_tts_all(jobs, VOICES[lang]))


async def _edge_tts_all(jobs, voice):
    """voice: one edge-tts voice id for all jobs, or a per-job list of voice ids."""
    n = len(jobs)
    per_job = isinstance(voice, (list, tuple))
    paths: list[str | None] = []
    for idx, (text, base) in enumerate(jobs, 1):
        text = (text or "").strip()
        v = voice[idx - 1] if per_job else voice
        path: str | None = None
        if text and v and re.search(r"[^\W_]", text, re.UNICODE):
            path = base + ".mp3"
            if os.path.exists(path) and os.path.getsize(path) > 0:
                paths.append(path)
                print(f"__PCT__ {12 + int(73 * idx / n)}", flush=True)
                continue
            for attempt in range(2):
                try:
                    await edge_tts.Communicate(text, v, rate=EDGE_TTS_RATE).save(path)
                    break
                except Exception as e:
                    if attempt == 0:
                        await asyncio.sleep(1.0)
                        continue
                    print("   skipped 1 segment:", str(e)[:50])
                    path = None
        paths.append(path)
        print(f"__PCT__ {12 + int(73 * idx / n)}", flush=True)
    return paths


def fit_segment(mp3: str, wav: str, slot_ms: int, ffmpeg: str,
                max_tempo: float = MAX_FIT_TEMPO) -> AudioSegment:
    """Load a spoken segment; if it overruns its time slot, speed it up (pitch kept).
    max_tempo caps the speed-up - kept low for XTTS so its natural speech stays clear."""
    seg = AudioSegment.from_file(mp3)
    if slot_ms > 0 and len(seg) / slot_ms > 1.03:
        tempo = min(len(seg) / slot_ms, max_tempo)
        run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-i", mp3, "-filter:a", f"atempo={tempo:.3f}", wav])
        if os.path.exists(wav):
            return AudioSegment.from_file(wav)
    return seg


# ---------------------------------------------------------------------------
# tone="original": make the dubbed voice track the ORIGINAL speaker's pace,
# pitch and loudness (still the same fixed AI voice - no cloning).
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def estimate_f0(seg: "AudioSegment") -> float | None:
    """Rough fundamental-frequency (Hz) of a voice slice via autocorrelation.
    Returns the median voiced F0, or None when nothing reliable is found
    (silence, music-heavy, numpy unavailable). Kept deliberately cheap."""
    if np is None or len(seg) < 80:
        return None
    s = seg.set_channels(1)
    if s.frame_rate > 8000:
        s = s.set_frame_rate(8000)
    sr = s.frame_rate
    x = np.asarray(s.get_array_of_samples(), dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0 or x.size < sr // 20:
        return None
    x = x / peak
    lag_min, lag_max = int(sr / 350.0), int(sr / 70.0)
    frame = int(sr * 0.04)
    hop = max(1, frame // 2)
    f0s: list[float] = []
    frames = 0
    for start in range(0, x.size - frame, hop):
        if frames >= 250:
            break
        frames += 1
        w = x[start:start + frame]
        if np.sqrt(np.mean(w * w)) < 0.05:        # energy gate: skip unvoiced
            continue
        w = w - np.mean(w)
        ac = np.correlate(w, w, mode="full")[frame - 1:]
        if ac[0] <= 0 or ac.size <= lag_max:
            continue
        lag = int(np.argmax(ac[lag_min:lag_max])) + lag_min
        if ac[lag] / ac[0] < 0.4:                 # voicing strength threshold
            continue
        # Octave guard: autocorrelation often peaks at 2x/3x the true period
        # (a sub-harmonic), reading the pitch an octave too low - which flips a
        # female line to "male". If half/third that lag still correlates nearly
        # as strongly, the real period is the shorter one.
        for div in (2, 3):
            sub = lag // div
            if sub >= lag_min and ac[sub] >= 0.8 * ac[lag]:
                lag = sub
                break
        f0s.append(sr / lag)
    if len(f0s) < 3:
        return None
    return float(np.median(f0s))


def load_orig_audio(video: str, vocals: str | None, work: str, base: str,
                    ffmpeg: str) -> "AudioSegment | None":
    """Original voice as one AudioSegment for analysis. Prefers the Demucs
    vocals stem (clean) when present, else a mono 16k extract of the video."""
    src = vocals if (vocals and os.path.exists(vocals)) else None
    if src is None:
        # No clean vocals: fall back to a mono 16k extract. Always re-extract from
        # THIS video rather than trusting a same-named leftover, so a re-uploaded
        # "download.mp4" can't analyse the previous clip's audio.
        src = os.path.join(work, base + "_orig16k.wav")
        run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-i", video, "-vn", "-ac", "1", "-ar", "16000", src])
        if not os.path.exists(src):
            return None
    try:
        return AudioSegment.from_file(src)
    except Exception:
        return None


def compute_orig_stats(orig: "AudioSegment | None",
                       segs: list[dict]) -> tuple[list[dict], float | None]:
    """Per-segment loudness (dBFS) and pitch (Hz) of the original, plus the
    median loudness used as the baseline for relative gain."""
    stats: list[dict] = []
    dbfs_vals: list[float] = []
    for s in segs:
        d: float | None = None
        f0: float | None = None
        if orig is not None:
            a = orig[int(s["start"] * 1000):int(s["end"] * 1000)]
            if len(a) > 0:
                dv = a.dBFS
                if dv != float("-inf") and dv == dv:   # exclude -inf / NaN
                    d = dv
                    dbfs_vals.append(dv)
                f0 = estimate_f0(a)
        stats.append({"dbfs": d, "f0": f0})
    if not dbfs_vals:
        median = None
    elif np is not None:
        median = float(np.median(dbfs_vals))
    else:
        median = sorted(dbfs_vals)[len(dbfs_vals) // 2]
    return stats, median


def fit_segment_expressive(src: str, out_wav: str, target_ms: int,
                           pitch_ratio: float, gain_db: float,
                           ffmpeg: str) -> "AudioSegment":
    """Stretch a spoken segment to the original's duration (pace), nudge its
    pitch toward the original speaker, and match relative loudness."""
    seg = AudioSegment.from_file(src)
    length = len(seg)
    if length <= 0 or target_ms <= 0:
        return seg + gain_db if gain_db else seg
    # Cap the speed-up tighter than the global ceiling: this path also pitch-shifts
    # the voice, and stacking a >1.5x tempo on top turns speech into the "can't make
    # it out" mush the user hit. Over-long lines instead drift and recover at the
    # next pause (see the overlay loop), which is far more intelligible.
    tempo = _clamp(length / target_ms, 0.7, 1.35)
    p = _clamp(pitch_ratio, 0.85, 1.18)
    if abs(tempo - 1.0) < 0.04 and abs(p - 1.0) < 0.02:
        return seg + gain_db if gain_db else seg
    a = _clamp(tempo / p, 0.5, 2.0)               # net tempo after pitch correction
    if abs(p - 1.0) < 0.02:
        af = f"atempo={a:.4f}"
    else:
        new_sr = max(1, int(round(seg.frame_rate * p)))
        af = f"asetrate={new_sr},aresample={seg.frame_rate},atempo={a:.4f}"
    rc = run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
              "-i", src, "-filter:a", af, out_wav])
    if rc == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav) > 0:
        out = AudioSegment.from_file(out_wav)
    else:
        out = seg
    return out + gain_db if gain_db else out


# MyMemory (the fallback) needs explicit locale codes; Google is happy with short codes + auto-detect.
_MM_TARGET = {"vi": "vi-VN", "id": "id-ID", "ms": "ms-MY", "es": "es-ES", "en": "en-US",
              "fr": "fr-FR", "de": "de-DE", "it": "it-IT", "pt": "pt-BR", "pl": "pl-PL",
              "tr": "tr-TR", "ru": "ru-RU", "nl": "nl-NL", "cs": "cs-CZ", "ar": "ar-SA",
              "zh-CN": "zh-CN", "hu": "hu-HU", "ko": "ko-KR", "ja": "ja-JP", "hi": "hi-IN"}
_MM_SOURCE = {"zh": "zh-CN", "ko": "ko-KR", "ja": "ja-JP", "en": "en-GB",
              "ms": "ms-MY", "th": "th-TH", "vi": "vi-VN", "id": "id-ID", "es": "es-ES"}


def _chunk(lines: list[str], max_chars: int, max_lines: int = 0) -> list[list[str]]:
    chunks: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for ln in lines:
        if cur and (size + len(ln) + 1 > max_chars
                    or (max_lines and len(cur) >= max_lines)):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        chunks.append(cur)
    return chunks


def _run_translator(tr, lines: list[str], max_chars: int) -> list[str] | None:
    """Translate every line with one engine. Returns a same-length list, or None if it fails outright."""
    out: list[str] = []
    failures = 0
    for chunk in _chunk(lines, max_chars):
        parts: list[str] | None = None
        for attempt in range(3):
            try:
                cand = (tr.translate("\n".join(chunk)) or "").split("\n")
                if len(cand) == len(chunk):
                    parts = cand
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        if parts is None:
            parts = []
            for ln in chunk:
                v = ""
                try:
                    v = tr.translate(ln) or ""
                except Exception:
                    failures += 1
                parts.append(v)
                time.sleep(0.2)
        out.extend(parts)
    return None if failures >= len(lines) else out


def _drop_untranslated(lines: list[str], lang: str) -> list[str]:
    """A target-language voice can't speak Chinese. If a line comes back still
    containing CJK characters (translation throttled and passed the source through),
    blank it so it is skipped rather than voiced as gibberish.
    Exempt CJK target languages (Chinese/Japanese/Korean) - their correct
    translations are *made of* these characters, so we must not blank them."""
    if lang.startswith(("zh", "ja", "ko")):
        return lines
    cjk = re.compile(r"[㐀-鿿぀-ヿ가-힯]")
    return ["" if (ln and cjk.search(ln)) else ln for ln in lines]


# --- DeepSeek (LLM) translation -------------------------------------------
# DeepSeek's API is text-only: it translates, but the dubbed VOICE still comes
# from Piper/edge-tts below. An LLM gives more natural, timing-aware translations
# than Google MT and sidesteps Google's per-request throttling. Key resolution:
# explicit arg (GUI field) > DEEPSEEK_API_KEY env var. deepseek-v4-flash is the
# cheapest/fastest model and is plenty for translation.
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"
_LANG_NAMES = {"vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
               "es": "Latin American Spanish", "en": "English",
               "fr": "French", "de": "German", "it": "Italian",
               "pt": "Brazilian Portuguese", "pl": "Polish", "tr": "Turkish",
               "ru": "Russian", "nl": "Dutch", "cs": "Czech", "ar": "Arabic",
               "zh-CN": "Simplified Chinese", "hu": "Hungarian", "ko": "Korean",
               "ja": "Japanese", "hi": "Hindi"}


def _deepseek_key(explicit: str = "") -> str:
    return (explicit or os.environ.get("DEEPSEEK_API_KEY", "")).strip()


def _deepseek_chat(messages: list[dict], api_key: str,
                   temperature: float = 1.0, max_tokens: int = 8192) -> str | None:
    """One OpenAI-compatible chat call to DeepSeek. Returns the reply text, or
    None on failure. max_tokens is set to DeepSeek's ceiling so a long batch reply
    isn't silently truncated into invalid JSON. Real errors (bad key, rate limit)
    are printed - they used to be swallowed, hiding why the dub fell back to the
    free engine."""
    import urllib.request
    import urllib.error
    body = json.dumps({"model": DEEPSEEK_MODEL, "messages": messages,
                       "temperature": temperature, "stream": False,
                       "max_tokens": max_tokens,
                       "response_format": {"type": "json_object"}}).encode("utf-8")
    req = urllib.request.Request(
        DEEPSEEK_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:160]
        except Exception:
            detail = ""
        print(f"  DeepSeek HTTP {e.code}: {detail}", flush=True)
        return None
    except Exception as e:
        print(f"  DeepSeek request error: {str(e)[:120]}", flush=True)
        return None


def _deepseek_chat_fn(api_key: str):
    """Return a plain chat(system, user) -> str|None callable bound to DeepSeek, for
    add-on modules (speech_rate length-fit) that just need 'send these two messages,
    get the reply'. _deepseek_chat already forces a JSON-object response, which the
    speech_rate prompts ask for; a cooler temperature keeps the rewrite faithful."""
    def _chat(system: str, user: str) -> str | None:
        return _deepseek_chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}], api_key, temperature=0.7)
    return _chat


# Keep each DeepSeek request small. A big batch (the old 3500-char chunk = ~130
# dense-Chinese lines) makes the model miscount its output lines OR emit invalid
# JSON (an unescaped quote inside a translation), and EITHER one used to throw the
# whole translation away and fall back to the free engine. ~40 lines / 1200 chars
# per request keeps the line count exact and the JSON valid.
_DS_MAX_LINES = 40
_DS_MAX_CHARS = 1200


def build_story_bible(lines: list[str], api_key: str) -> dict | None:
    """Infer story context once so translation can resolve Chinese pronouns.

    Chinese drama recaps often use 他/她/TA loosely or inconsistently in captions.
    Translating small batches without a global cast list makes the model flip gender
    mid-story. This lightweight "bible" is cached per script and passed to every
    translation batch as context.
    """
    key = _deepseek_key(api_key)
    if not key or not lines:
        return None
    joined = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
    if len(joined) > 24000:
        # Keep prompt size bounded; first/middle/end is enough for cast inference.
        head = lines[:80]
        mid0 = max(80, len(lines) // 2 - 40)
        mid = lines[mid0:mid0 + 80]
        tail = lines[-80:]
        sample = head + ["..."] + mid + ["..."] + tail
        joined = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(sample))
    sys_msg = (
        "You are a story analyst for Chinese drama dubbing. Read the whole subtitle "
        "script and infer a compact translation bible. Chinese captions may use 他, 她, "
        "TA, or the wrong homophone/character inconsistently; infer the actual gender "
        "from names, relationships, plot facts, titles, and repeated actions. Do not "
        "guess wildly: mark unknown when uncertain. Reply with ONLY JSON containing: "
        '{"summary":"...","characters":[{"name":"...","aliases":["..."],'
        '"gender":"M|F|unknown","role":"...","relationships":"..."}],'
        '"pronoun_notes":["..."],"translation_rules":["..."]}.')
    content = _deepseek_chat(
        [{"role": "system", "content": sys_msg},
         {"role": "user", "content": joined}], key, temperature=0.2)
    if not content:
        return None
    try:
        data = json.loads(content)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    chars = data.get("characters")
    if not isinstance(chars, list):
        chars = []
    data["characters"] = chars[:12]
    data["summary"] = str(data.get("summary", ""))[:600]
    data["pronoun_notes"] = [str(x)[:180] for x in (data.get("pronoun_notes") or [])[:12]]
    data["translation_rules"] = [str(x)[:180] for x in (data.get("translation_rules") or [])[:12]]
    return data


def _story_bible_brief(story_bible: dict | None) -> str:
    if not story_bible:
        return ""
    parts = []
    if story_bible.get("summary"):
        parts.append("Story summary: " + str(story_bible["summary"]))
    chars = []
    for c in story_bible.get("characters") or []:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "")).strip()
        gender = str(c.get("gender", "unknown")).strip()
        role = str(c.get("role", "")).strip()
        aliases = ", ".join(str(a) for a in (c.get("aliases") or [])[:4])
        rel = str(c.get("relationships", "")).strip()
        chars.append(f"{name or 'unknown'} ({gender}; {role}; aliases: {aliases}; {rel})")
    if chars:
        parts.append("Characters: " + " | ".join(chars[:12]))
    notes = story_bible.get("pronoun_notes") or []
    if notes:
        parts.append("Pronoun/gender notes: " + " | ".join(str(n) for n in notes[:12]))
    rules = story_bible.get("translation_rules") or []
    if rules:
        parts.append("Rules: " + " | ".join(str(r) for r in rules[:8]))
    return "\n".join(parts)[:3500]


def _feedback_lessons(limit: int = 10) -> list[str]:
    try:
        store = FeedbackStorage(os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback"))
        return store.lessons(limit=limit)
    except Exception:
        return []


def _merge_feedback_lessons(story_bible: dict | None) -> dict | None:
    lessons = _feedback_lessons()
    if not lessons:
        return story_bible
    data = dict(story_bible or {})
    rules = list(data.get("translation_rules") or [])
    for lesson in lessons:
        rule = "Human correction memory: " + lesson
        if rule not in rules:
            rules.append(rule)
    data["translation_rules"] = rules
    return data


def _visual_context_brief(visual_contexts: list[dict] | None) -> str:
    if not visual_contexts:
        return ""
    lines = []
    for idx, ctx in enumerate(visual_contexts[:40], 1):
        if not isinstance(ctx, dict):
            continue
        bits = []
        for key in ("setting", "speaker_expression", "visual_context"):
            val = str(ctx.get(key) or "").strip()
            if val:
                bits.append(f"{key}: {val}")
        terms = ctx.get("ambiguous_terms")
        if isinstance(terms, dict) and terms:
            bits.append("ambiguous_terms: " + json.dumps(terms, ensure_ascii=False))
        if bits:
            lines.append(f"{idx}. " + "; ".join(bits))
    if not lines:
        return ""
    return (
        "Visual context by line. Use it to resolve ambiguous words, emotion, tone, "
        "and whether a phrase should sound formal, casual, joking, angry, or urgent:\n"
        + "\n".join(lines)
    )[:3500]


def _ds_span(lines: list[str], target: str, api_key: str,
             durations: list[float] | None, depth: int = 0,
             story_bible: dict | None = None,
             visual_contexts: list[dict] | None = None) -> list[str | None]:
    """Translate one batch of lines via DeepSeek, returning a same-length list
    (None for any line we couldn't get). On a bad reply (wrong line count, or
    JSON the model malformed) we retry at a cooler temperature; if it still fails
    and the batch has more than one line, we SPLIT it and translate the halves -
    smaller batches are far more reliable - instead of discarding everything and
    dropping to the free engine."""
    if not lines:
        return []
    if durations is not None:
        numbered = "\n".join(f"{i + 1}. ({durations[i]:.1f}s) {ln}"
                             for i, ln in enumerate(lines))
        budget_rule = (
            "Each line is prefixed with its spoken-time budget in seconds, "
            "like (2.3s). Your translation MUST be short enough to say "
            "naturally within that budget - tighten phrasing and cut filler "
            "rather than overrun, while keeping the core meaning. ")
    else:
        numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
        budget_rule = ("Keep each line about as short as the source so it "
                       "fits the original speaking time. ")
    bible = _story_bible_brief(story_bible)
    visual = _visual_context_brief(visual_contexts)
    context_rule = (
        "Use the story context below to keep character gender, pronouns, names, "
        "relationships, and who-did-what consistent across lines. For Chinese 他/她/TA, "
        "do NOT translate mechanically; infer whether the referent is he/she/they from "
        "the character bible and local context. If uncertain, use the character name or "
        "a neutral phrasing rather than the wrong gender.\n\n" + bible + "\n\n"
        if bible else
        "Keep character gender, pronouns, names, relationships, and who-did-what "
        "consistent across lines. For Chinese 他/她/TA, infer the referent from story "
        "context; if uncertain, use the character name or neutral phrasing.\n")
    if visual:
        context_rule += "\n" + visual + "\n"
    sys_msg = (
        f"You are a professional video-dubbing translator. Translate each "
        f"numbered line into natural, spoken {target}, the way a voice actor "
        f"would say it. {context_rule}{budget_rule}Translate fully - never keep or "
        f"transliterate the source language. Reply with ONLY a JSON object "
        f'{{"lines": [...]}} holding exactly {len(lines)} strings, in order, '
        f"one per input line.")
    for temp in (1.0, 0.7, 0.4):
        content = _deepseek_chat(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": numbered}], api_key, temperature=temp)
        if not content:
            continue
        try:
            parts = json.loads(content).get("lines")
        except Exception:
            parts = None
        if isinstance(parts, list) and len(parts) == len(lines):
            return ["" if p is None else str(p) for p in parts]
    # Still no clean reply: split and retry the halves (down to single lines).
    if len(lines) > 1 and depth < 8:
        mid = len(lines) // 2
        dl = durations[:mid] if durations is not None else None
        dr = durations[mid:] if durations is not None else None
        vl = visual_contexts[:mid] if visual_contexts is not None else None
        vr = visual_contexts[mid:] if visual_contexts is not None else None
        return (_ds_span(lines[:mid], target, api_key, dl, depth + 1, story_bible, vl)
                + _ds_span(lines[mid:], target, api_key, dr, depth + 1, story_bible, vr))
    return [None] * len(lines)


def _deepseek_translate(lines: list[str], lang: str, api_key: str,
                        durations: list[float] | None = None,
                        story_bible: dict | None = None,
                        visual_contexts: list[dict] | None = None) -> list[str] | None:
    """Translate every line with DeepSeek. Returns a same-length list (a line that
    couldn't be translated even after splitting becomes ""), or None only when the
    whole call fails outright (e.g. bad key) so the caller can fall back.
    durations: per-line spoken-time budget (seconds). When given, each line is
    tagged with its budget and the model is told to keep the translation short
    enough to say in that time - this is what keeps the dub from drifting out of
    sync when a translation would otherwise run longer than the original line."""
    target = _LANG_NAMES.get(lang, lang)
    out: list[str | None] = []
    pos = 0
    for chunk in _chunk(lines, _DS_MAX_CHARS, _DS_MAX_LINES):
        d = durations[pos:pos + len(chunk)] if durations is not None else None
        vc = visual_contexts[pos:pos + len(chunk)] if visual_contexts is not None else None
        out.extend(_ds_span(chunk, target, api_key, d, story_bible=story_bible,
                            visual_contexts=vc))
        pos += len(chunk)
    if all(v is None for v in out):        # nothing came back -> let caller fall back
        return None
    return ["" if v is None else v for v in out]


def cleanup_script_lines(lines: list[str], api_key: str) -> tuple[list[str], int]:
    """Correct STT glitches before translation while preserving line count/order."""
    key = _deepseek_key(api_key)
    if not key or not lines:
        return list(lines), 0
    out = list(lines)
    changed = 0
    pos = 0
    for chunk in _chunk(lines, _DS_MAX_CHARS, _DS_MAX_LINES):
        numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(chunk))
        sys_msg = (
            "You are a meticulous subtitle transcription editor. Fix obvious "
            "speech-to-text mistakes, punctuation, spacing, and wrong characters "
            "without changing meaning or language. Preserve names when uncertain. "
            f"Reply with ONLY a JSON object {{\"lines\": [...]}} holding exactly "
            f"{len(chunk)} strings, one per input line, in order.")
        content = _deepseek_chat(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": numbered}], key, temperature=0.2)
        try:
            fixed = json.loads(content or "{}").get("lines")
        except Exception:
            fixed = None
        if isinstance(fixed, list) and len(fixed) == len(chunk):
            for j, val in enumerate(fixed):
                text = str(val or "").strip()
                if text and text != out[pos + j]:
                    out[pos + j] = text
                    changed += 1
        pos += len(chunk)
    return out, changed


def translate_segments(lines: list[str], lang: str, source_lang: str = "auto",
                       api_key: str = "",
                       durations: list[float] | None = None,
                       story_bible: dict | None = None,
                       visual_contexts: list[dict] | None = None) -> list[str]:
    """Translate lines: DeepSeek first (when a key is set), then Google, then
    MyMemory. Batched to avoid the throttling hundreds of separate requests
    trigger. A line that cannot be translated becomes "" (skipped later) - never
    the original text, which a target-language voice cannot speak.
    durations: optional per-line spoken-time budget (seconds) passed to DeepSeek
    so translations stay short enough to keep the dub in sync.
    """
    if not lines:
        return []
    key = _deepseek_key(api_key)
    if key:
        res = _deepseek_translate(lines, lang, key, durations, story_bible, visual_contexts)
        if res:
            ok = _drop_untranslated(res, lang)
            if sum(1 for r in ok if r.strip()) >= len(lines) * 0.6:
                print(f"  [{lang}] translated via DeepSeek", flush=True)
                return ok
        print("  DeepSeek translate failed; falling back to free engine", flush=True)
    try:
        res = _run_translator(GoogleTranslator(source="auto", target=lang), lines, 4500)
        if res:
            ok = _drop_untranslated(res, lang)
            # Reject a throttled reply that just echoed the source back untranslated;
            # fall through to MyMemory instead of voicing Chinese.
            if sum(1 for r in ok if r.strip()) >= len(lines) * 0.6:
                return ok
    except Exception:
        pass
    src = _MM_SOURCE.get(source_lang, source_lang)
    tgt = _MM_TARGET.get(lang, lang)
    try:
        res = _run_translator(MyMemoryTranslator(source=src, target=tgt), lines, 480)
        if res:
            ok = _drop_untranslated(res, lang)
            if any(r.strip() for r in ok):
                return ok
    except Exception:
        pass
    return [""] * len(lines)


def _safe_name(text: str) -> str:
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", text)
    text = re.sub(r"\s+", " ", text).strip().strip(".")
    return text[:40].strip()


def _unique_out(out_dir: str, name: str, lang: str) -> str:
    p = os.path.join(out_dir, f"{name}_{lang}.mp4")
    i = 2
    while os.path.exists(p):
        p = os.path.join(out_dir, f"{name}_{lang}_{i}.mp4")
        i += 1
    return p


def classify_genders(orig_stats: list[dict] | None, n: int) -> list[str]:
    """Label each of n segments 'M' or 'F' from its original-voice pitch (F0).
    A line with no reliable F0 inherits the previous labelled line (speech tends to
    stay with one speaker), else the video's dominant gender, else 'F'."""
    raw: list[str | None] = []
    for i in range(n):
        st = orig_stats[i] if (orig_stats and i < len(orig_stats)) else None
        f0 = st.get("f0") if st else None
        raw.append(("M" if f0 < GENDER_F0_SPLIT else "F") if f0 else None)
    known = [g for g in raw if g]
    dominant = "M" if known.count("M") > known.count("F") else "F"
    out: list[str] = []
    last = dominant
    for g in raw:
        if g:
            last = g
        out.append(g or last)
    return out


def ai_assign_cast(segs: list[dict], texts: list[str], api_key: str,
                   max_speakers: int = 8,
                   story_bible: dict | None = None) -> tuple[list[int], list[str], list[str]] | None:
    """Ask DeepSeek to group script lines by likely character/narrator.

    Audio diarization is useful, but drama recap videos often mix narrator speech with
    short in-scene dialogue, and similar same-gender voices collapse together. This
    text-aware pass lets the model infer "narrator", "male lead", "mother", etc. from
    the script and assign each line a stable cast id plus a coarse gender. The returned
    ids drive per-character voice selection; genders pick male/female voice pools.
    """
    key = _deepseek_key(api_key)
    n = len(segs)
    if not key or not n or len(texts) != n:
        return None
    lines = []
    for i, s in enumerate(segs):
        src = str(s.get("text", "")).replace("\n", " ").strip()
        tr = str(texts[i] or "").replace("\n", " ").strip()
        lines.append(f"{i + 1}. SRC: {src}\n   DUB: {tr}")
    user = "\n".join(lines)
    if len(user) > 18000:
        # Long scripts are still handled by audio diarization; keeping this bounded
        # avoids slow/fragile giant JSON replies.
        return None
    bible = _story_bible_brief(story_bible)
    sys_msg = (
        "You are a casting director for video dubbing. Group each numbered subtitle "
        "line by who is speaking. Use speaker 0 for the narrator/recap voice when a "
        "line is narration. Use speaker 1, 2, 3... for recurring on-screen characters. "
        "Keep the same speaker id for the same character across the whole script. "
        f"Use at most {max_speakers} speakers total. Also assign gender M or F for "
        "the speaking role. Use the story bible to resolve Chinese 他/她/TA and avoid "
        "flipping a character's gender. Use M for unknown narrator unless the line "
        "clearly reads female. "
        + (f"\n\nStory bible:\n{bible}\n\n" if bible else "")
        + "Reply with ONLY a JSON object like "
        '{"lines":[{"speaker":0,"gender":"M","role":"narrator"}]} with exactly one '
        "entry per input line, in order.")
    content = _deepseek_chat(
        [{"role": "system", "content": sys_msg},
         {"role": "user", "content": user}], key, temperature=0.2)
    if not content:
        return None
    try:
        arr = json.loads(content).get("lines")
    except Exception:
        return None
    if not isinstance(arr, list) or len(arr) != n:
        return None
    labels: list[int] = []
    genders: list[str] = []
    roles: list[str] = []
    remap: dict[int, int] = {}
    for item in arr:
        if not isinstance(item, dict):
            return None
        try:
            raw = int(item.get("speaker", 0))
        except Exception:
            raw = 0
        raw = max(0, min(max_speakers - 1, raw))
        if raw not in remap:
            remap[raw] = len(remap)
        labels.append(remap[raw])
        g = str(item.get("gender", "M")).upper()
        genders.append("F" if g.startswith("F") else "M")
        role = re.sub(r"\s+", " ", str(item.get("role", "")).strip().lower())
        roles.append(role[:40] or f"speaker {labels[-1] + 1}")
    if len(set(labels)) <= 1:
        return None
    print(f"  AI cast: {len(set(labels))} characters/narrator voices", flush=True)
    return labels, genders, roles


def build_scene_fx(vocals_path: str, segs: list[dict], total_ms: int,
                   pad_ms: int = 150, gain_db: float = -9.0) -> "AudioSegment":
    """Recover scene vocal SFX (panting, grunts, fight shouts) from the original
    vocals stem WITHOUT the original dialogue: keep the stem only in the gaps
    between transcribed speech spans, silence the speech spans (where the source
    language sits), and duck the result so it sits under the dub."""
    src = AudioSegment.from_file(vocals_path)
    if len(src) < total_ms:
        src = src + AudioSegment.silent(total_ms - len(src))
    src = src[:total_ms]
    spans: list[list[int]] = []
    for s in segs:
        a = max(0, int(s["start"] * 1000) - pad_ms)
        b = min(total_ms, int(s["end"] * 1000) + pad_ms)
        if b > a:
            spans.append([a, b])
    spans.sort()
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = AudioSegment.empty()
    cursor = 0
    for a, b in merged:
        if a > cursor:
            out += src[cursor:a]
        out += AudioSegment.silent(b - a)
        cursor = b
    if cursor < total_ms:
        out += src[cursor:total_ms]
    return out + gain_db


def duck_music_under_voice(music: "AudioSegment", voice: "AudioSegment",
                           speech_duck_db: float = -16.0,
                           gap_duck_db: float = -8.0,
                           step_ms: int = 80) -> "AudioSegment":
    """Return a music bed that is lower when dubbed voice is present.

    Fixed ducking makes quiet gaps feel lifeless and still lets loud music mask
    speech. This cheap envelope follows the rendered voice track in small chunks.
    """
    total = min(len(music), len(voice))
    out = AudioSegment.empty()
    floor = -42.0
    for pos in range(0, total, step_ms):
        m = music[pos:pos + step_ms]
        v = voice[pos:pos + step_ms]
        gain = speech_duck_db if (v.dBFS != float("-inf") and v.dBFS > floor) else gap_duck_db
        out += m + gain
    if len(music) > total:
        out += music[total:] + gap_duck_db
    return out


def build_speaker_refs(vocals: str, segs: list[dict], labels: list[int],
                       work_dir: str) -> dict[int, str]:
    """For each detected speaker, stitch up to ~20s of THEIR isolated-voice segments
    into one reference wav. XTTS clones from these so every speaker keeps their own
    voice. Returns {speaker_id: ref_wav_path} (only speakers with usable audio)."""
    refs: dict[int, str] = {}
    try:
        clip = AudioSegment.from_file(vocals)
    except Exception:
        return refs
    buckets: dict[int, list[dict]] = {}
    for s, lab in zip(segs, labels):
        buckets.setdefault(lab, []).append(s)
    for lab, items in buckets.items():
        comb = AudioSegment.silent(duration=0)
        for s in items:
            comb += clip[int(s["start"] * 1000):int(s["end"] * 1000)]
            if comb.duration_seconds >= 20:
                break
        if comb.duration_seconds < 1.5:        # too little to model a voice from
            continue
        p = os.path.join(work_dir, f"spk_{lab}.wav")
        try:
            comb.export(p, format="wav")
            refs[lab] = p
        except Exception:
            continue
    return refs


# EBU R128 two-pass loudness (upgrade of the single-pass loudnorm we used at the
# three render sites). Single-pass `loudnorm` normalises dynamically in one go, which
# can pump or limit unevenly; measuring the finished mix first and then applying with
# the measured stats + linear=true gives an accurate, transparent normalisation to the
# same -14 LUFS / -1.5 dBTP target. Falls back to single-pass automatically when the
# measure pass fails, and can be forced off with env DUB_SINGLEPASS_LOUDNORM=1.
_LN_I, _LN_TP, _LN_LRA = -14.0, -1.5, 11.0


def _loudnorm_single(I: float = _LN_I, TP: float = _LN_TP, LRA: float = _LN_LRA) -> str:
    return f"loudnorm=I={I}:TP={TP}:LRA={LRA}"


def _measure_loudnorm(ffmpeg: str, wav: str, I: float = _LN_I, TP: float = _LN_TP,
                      LRA: float = _LN_LRA) -> dict | None:
    """First pass: measure `wav`'s integrated loudness/true-peak/range via
    loudnorm:print_format=json. Returns the measured_* inputs, or None on ANY failure
    (missing file, non-zero rc, timeout, unparseable output) so the caller falls back
    to a plain single-pass filter. Skipped entirely if DUB_SINGLEPASS_LOUDNORM=1."""
    if os.environ.get("DUB_SINGLEPASS_LOUDNORM", "") == "1":
        return None
    if not wav or not os.path.exists(wav):
        return None
    filt = f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json"
    try:
        p = subprocess.run([ffmpeg, "-hide_banner", "-nostats", "-i", wav,
                            "-af", filt, "-f", "null", "-"],
                           capture_output=True, text=True, timeout=600)
    except Exception:
        return None
    err = p.stderr or ""
    i, j = err.rfind("{"), err.rfind("}")        # JSON block is printed last on stderr
    if i < 0 or j <= i:
        return None
    try:
        m = json.loads(err[i:j + 1])
        return {k: m[k] for k in ("input_i", "input_tp", "input_lra",
                                  "input_thresh", "target_offset")}
    except Exception:
        return None


def _loudnorm_filter(ffmpeg: str, wav: str, I: float = _LN_I, TP: float = _LN_TP,
                     LRA: float = _LN_LRA) -> str:
    """The loudnorm filter string to apply to `wav`: a measured, linear two-pass
    filter when the measure pass succeeds, else the plain single-pass filter."""
    m = _measure_loudnorm(ffmpeg, wav, I, TP, LRA)
    if not m:
        return _loudnorm_single(I, TP, LRA)
    print("  loudness: two-pass (measured) normalisation", flush=True)
    return (f"loudnorm=I={I}:TP={TP}:LRA={LRA}:"
            f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
            f"offset={m['target_offset']}:linear=true")


def make_dub(lang: str, segs: list[dict], total_ms: int, bed: str | None,
             video: str, out_dir: str, base: str, work: str, ffmpeg: str,
             source_lang: str = "auto", cover_subs: bool = False, subs_band: int = 18,
             cover_region: str = "", fb_vertical: bool = False,
             make_srt: bool = False, naming: str = "source", burn_subs: bool = False,
             tone: str = "natural", orig_stats: list[dict] | None = None,
             median_dbfs: float | None = None, audio_only: bool = False,
             clone: bool = False, vocals: str | None = None,
             api_key: str = "", gender_mode: bool = False,
             scene_fx: bool = False, spk_labels: list[int] | None = None,
             texts_override: list[str] | None = None,
             length_fit: bool = False, cast_mode: bool = False,
             script_cleanup: bool = False, preset: str = "",
             rights_mode: str = "", voice_duck: bool = False,
             cast_roles: list[str] | None = None,
             cast_genders: list[str] | None = None,
             story_bible: dict | None = None) -> None:
    if naming != "firstline":
        out_mp4 = os.path.join(out_dir, f"{base}_{lang}.mp4")
        if os.path.exists(out_mp4):
            print(f"  [{lang}] already done; skipping")
            return

    # Re-dub path: the editor passes the human-edited translations, so skip the
    # translator entirely and voice exactly what the user approved.
    if texts_override is not None and len(texts_override) == len(segs):
        texts = list(texts_override)
        print(f"__STAGE__ {lang.upper()}: using edited script", flush=True)
    else:
        print(f"__STAGE__ {lang.upper()}: translating", flush=True)
        src_lines = [s["text"] for s in segs]
        if script_cleanup:
            print(f"__STAGE__ {lang.upper()}: cleaning script", flush=True)
            cleaned, nclean = cleanup_script_lines(src_lines, api_key)
            if nclean:
                print(f"  script cleanup adjusted {nclean}/{len(src_lines)} lines", flush=True)
                src_lines = cleaned
                for s, txt in zip(segs, cleaned):
                    s["text"] = txt
        # Budget each translation to the original line's spoken time so DeepSeek keeps
        # it short enough to stay in sync (floor avoids an impossible budget on blips).
        durations = [max(0.6, float(s.get("visual_budget") or (s["end"] - s["start"]))) for s in segs]
        visual_contexts = [s.get("visual_context") or {} for s in segs]
        if story_bible is None and _deepseek_key(api_key):
            bible_sig = hashlib.sha1(json.dumps(src_lines, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
            bible_path = os.path.join(work, "analysis", f"{base}_{bible_sig}_story.json")
            if os.path.exists(bible_path):
                try:
                    story_bible = json.load(open(bible_path, encoding="utf-8"))
                    print("  story context: reused cached character bible", flush=True)
                except Exception:
                    story_bible = None
            if story_bible is None:
                print(f"__STAGE__ {lang.upper()}: understanding story", flush=True)
                story_bible = build_story_bible(src_lines, api_key)
                if story_bible:
                    try:
                        json.dump(story_bible, open(bible_path, "w", encoding="utf-8"),
                                  ensure_ascii=False)
                    except Exception:
                        pass
                    print(f"  story context: {len(story_bible.get('characters') or [])} characters", flush=True)
        story_bible = _merge_feedback_lessons(story_bible)
        tr_sig = hashlib.sha1(json.dumps(
            {"v": 2, "lang": lang, "src": src_lines,
             "story": story_bible or {},
             "visual": visual_contexts,
             "dur": [round(d, 2) for d in durations]},
            ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        tr_cache = os.path.join(work, "analysis", f"{base}_{lang}_{tr_sig}_tr.json")
        texts = None
        if os.path.exists(tr_cache):
            try:
                cached_tr = json.load(open(tr_cache, encoding="utf-8")).get("texts")
                if isinstance(cached_tr, list) and len(cached_tr) == len(src_lines):
                    texts = [str(x or "") for x in cached_tr]
                    print(f"  [{lang}] reused cached translation", flush=True)
            except Exception:
                texts = None
        if texts is None:
            texts = translate_segments(src_lines, lang, source_lang, api_key, durations,
                                       story_bible, visual_contexts)
            try:
                os.makedirs(os.path.dirname(tr_cache), exist_ok=True)
                json.dump({"texts": texts}, open(tr_cache, "w", encoding="utf-8"),
                          ensure_ascii=False)
            except Exception:
                pass
        # Optional second-pass length-fit (OmniVoice speech_rate port): dub.py already
        # hands DeepSeek each line's time budget during translation, but the model
        # doesn't always obey. This pass estimates each translated line's reading time
        # from a per-language chars/sec table and asks DeepSeek to trim/expand ONLY the
        # lines that still overshoot/undershoot their slot - so the audio fit step below
        # has to time-stretch far less. Needs a DeepSeek key; a no-op without one. Not
        # run on the re-dub path (texts_override) - the user already approved those.
        if length_fit:
            key = _deepseek_key(api_key)
            if key:
                import speech_rate
                slots = [max(0.6, float(s.get("visual_budget") or (s["end"] - s["start"]))) for s in segs]
                srcs = [s.get("text", "") for s in segs]
                print(f"__STAGE__ {lang.upper()}: fitting length", flush=True)
                texts, nfit = speech_rate.fit_texts_counted(
                    texts, slots, lang, _deepseek_chat_fn(key), srcs)
                print(f"  [{lang}] length-fit adjusted {nfit}/{len(texts)} lines",
                      flush=True)
        key = _deepseek_key(api_key)
        texts, align_decisions = enforce_alignment_constraints(
            segs,
            texts,
            lang,
            _deepseek_chat_fn(key) if key else None,
            max_tempo=MAX_FIT_TEMPO,
        )
        n_hard = sum(1 for d in align_decisions if d.action not in ("accept", "time_stretch"))
        n_stretch = sum(1 for d in align_decisions if d.action == "time_stretch")
        if n_hard or n_stretch:
            print(f"  [{lang}] visual alignment: {n_stretch} stretch, {n_hard} truncation decisions",
                  flush=True)
    print("__PCT__ 12", flush=True)

    if naming == "firstline":
        first = next((t for t in texts if t.strip()), "")
        out_mp4 = _unique_out(out_dir, _safe_name(first) or base, lang)

    if make_srt:
        write_srt(os.path.splitext(out_mp4)[0] + ".srt", segs, texts)

    seg_dir = os.path.join(work, f"tts_{base}_{lang}")
    os.makedirs(seg_dir, exist_ok=True)
    jobs = [(texts[i], _seg_base(seg_dir, i, texts[i])) for i in range(len(texts))]
    seg_voices = None
    genders = None
    # Multi-speaker: spk_labels (computed once per video by run_one's diarize_once
    # and reused across languages) groups the lines by who's talking, so each
    # speaker gets a distinct voice (and, with cloning, their OWN voice). None when
    # there's a single speaker or diarization was unavailable.
    spk_refs = None
    # Classify M/F from the original pitch whenever gender mode is on - even when
    # speakers were also split. Previously this lived in an `elif`, so detecting
    # speakers silently disabled the gender choice; now the two combine (and `genders`
    # is also handed to XTTS / the pitch step below).
    ai_genders = None
    ai_roles = None
    if cast_mode:
        cast_sig = hashlib.sha1(json.dumps(
            [[s.get("start"), s.get("end"), s.get("text"), texts[i] if i < len(texts) else ""]
             for i, s in enumerate(segs)],
            ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        cast_path = os.path.join(seg_dir, f"_cast_{cast_sig}.json")
        cast = None
        if os.path.exists(cast_path):
            try:
                c = json.load(open(cast_path, encoding="utf-8"))
                cast = (c["labels"], c["genders"], c["roles"])
                print("  AI cast: reused cached character assignment", flush=True)
            except Exception:
                cast = None
        if cast is None:
            cast = ai_assign_cast(segs, texts, api_key, story_bible=story_bible)
            if cast:
                try:
                    json.dump({"labels": cast[0], "genders": cast[1], "roles": cast[2]},
                              open(cast_path, "w", encoding="utf-8"), ensure_ascii=False)
                except Exception:
                    pass
        if cast:
            spk_labels, ai_genders, ai_roles = cast
    if ai_genders is not None:
        genders = ai_genders
        cast_genders = ai_genders
        cast_roles = ai_roles
    elif gender_mode:
        genders = classify_genders(orig_stats, len(jobs))
        cast_genders = genders
    if spk_labels is not None:
        if genders is not None:
            seg_voices = _speaker_gender_voices(lang, spk_labels, genders)
            nm = sum(1 for g in genders if g == "M")
            print(f"  [{lang}] {len(set(spk_labels))} speakers, gender-matched "
                  f"({nm} male / {len(genders) - nm} female lines)", flush=True)
        else:
            pool = _speaker_voices(lang)
            seg_voices = [pool[spk_labels[i] % len(pool)] for i in range(len(jobs))]
    elif genders is not None and lang in GENDER_VOICES:
        seg_voices = [GENDER_VOICES[lang][g] for g in genders]
        nm = sum(1 for g in genders if g == "M")
        print(f"  [{lang}] gender voices: {nm} male / {len(genders) - nm} female lines", flush=True)
    print(f"__STAGE__ {lang.upper()}: voicing", flush=True)
    # XTTS-supported languages: prefer XTTS v2 (local, natural, and clones the
    # original speaker from the vocals stem when one is present). It supersedes Piper
    # AND the OpenVoice clone step; when coqui-tts / the model aren't installed
    # available() is False and we fall back to the normal Piper/edge-tts path.
    # Languages XTTS can't model (vi/id/ms) stay on edge-tts. See xtts.XTTS_LANGS.
    import xtts
    force_edge_voices = bool(seg_voices) or (not clone and lang in VOICES)
    use_xtts = (not force_edge_voices) and lang in xtts.XTTS_LANGS and xtts.available()
    if use_xtts:
        # With multi-speaker + clone, give XTTS one reference per speaker so each is
        # voiced in their OWN voice; without clone, XTTS assigns a distinct built-in
        # speaker per id. Otherwise: clone the single original (only when ticked), or
        # use the gender/built-in path. `vocals` can exist for unrelated reasons
        # (gender / scene-fx / keep-music all run Demucs), so we never clone by default.
        if spk_labels is not None and clone and vocals:
            spk_refs = build_speaker_refs(vocals, segs, spk_labels, seg_dir)
        paths = xtts.synthesize(jobs, vocals if (clone and spk_labels is None) else None,
                                genders, lang, spk_labels=spk_labels, spk_refs=spk_refs)
    else:
        paths = tts_all(jobs, lang, seg_voices)
    if clone and vocals and not use_xtts and spk_labels is None:
        # (OpenVoice clones to ONE reference, which would collapse multi-speaker voices
        # back into a single voice, so we skip it when speakers were split.)
        print(f"__STAGE__ {lang.upper()}: cloning original voice", flush=True)
        import voiceclone
        paths = voiceclone.clone_segments(paths, vocals, seg_dir, lang, ffmpeg)

    # Place each dubbed line WITHOUT overlapping the previous one. We anchor to the
    # original start, but if a long line overran, the next line waits until the prior
    # one truly ends (cursor). Any silent gap in the source lets `start` snap back to
    # the original time, so drift recovers at pauses instead of snowballing all video.
    canvas = AudioSegment.silent(duration=total_ms)
    cursor = 0
    for i, s in enumerate(segs):
        path = paths[i]
        if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        orig_start = int(s["start"] * 1000)
        # Keep the dub close to the source timing. An over-long translation can
        # push following lines only a little; otherwise the whole clip drifts late.
        start = max(orig_start, min(cursor, orig_start + MAX_SYNC_DRIFT_MS))
        nxt = int(segs[i + 1]["start"] * 1000) if i + 1 < len(segs) else total_ms
        visual_end = int(float(s.get("visual_end") or 0.0) * 1000)
        scene_end = int(float(s.get("scene_visual_end") or 0.0) * 1000)
        hard_end = min([x for x in (nxt, visual_end, scene_end) if x > start] or [nxt])
        room = max(300, hard_end - start)                      # physical visual boundary
        fit_wav = os.path.join(seg_dir, f"{i:03d}_fit.wav")
        if use_xtts:
            # XTTS speech is already natural but runs a bit longer than the source.
            # Pitch-warping (tone=original) or hard time-compression mangles neural
            # speech into the "can't understand it" mush, so for XTTS we do neither:
            # no pitch shift, only a mild speed-up, and let the timeline drift and
            # recover at the next pause instead of cramming every line into its slot.
            seg_audio = fit_segment(path, fit_wav, room, ffmpeg, max_tempo=1.15)
        elif tone == "original":
            st = orig_stats[i] if (orig_stats and i < len(orig_stats)) else None
            f0 = st.get("f0") if st else None
            ref = VOICE_REF_F0_MALE if (genders and genders[i] == "M") else VOICE_REF_F0
            pitch_ratio = (f0 / ref) if f0 else 1.0
            if st and st.get("dbfs") is not None and median_dbfs is not None:
                gain_db = _clamp(st["dbfs"] - median_dbfs, -6.0, 6.0)
            else:
                gain_db = 0.0
            seg_audio = fit_segment_expressive(path, fit_wav, room,
                                               pitch_ratio, gain_db, ffmpeg)
        else:
            seg_audio = fit_segment(path, fit_wav, room, ffmpeg)
        canvas = canvas.overlay(seg_audio, position=start)
        cursor = start + len(seg_audio)

    voice_track = canvas - 1
    if bed and os.path.exists(bed):
        music = AudioSegment.from_file(bed)
        if len(music) < total_ms:
            music = music + AudioSegment.silent(total_ms - len(music))
        # Duck the kept music a little more (was -6 dB): loud beds were masking the
        # dub and making the new voice hard to follow.
        music = music[:total_ms]
        if voice_duck:
            music = duck_music_under_voice(music, voice_track)
        else:
            music = music - 12
        mixed = music.overlay(voice_track)
    else:
        mixed = voice_track

    if scene_fx and vocals and os.path.exists(vocals):
        print(f"  [{lang}] keeping scene sounds (panting/shouts) from non-dialogue gaps", flush=True)
        mixed = mixed.overlay(build_scene_fx(vocals, segs, total_ms))

    mix_wav = os.path.join(work, f"{base}_{lang}_mix.wav")
    mixed.export(mix_wav, format="wav")
    # Measure the finished mix once and reuse the (two-pass) loudnorm filter at
    # whichever render site runs below - auto-falls back to single-pass on failure.
    ln = _loudnorm_filter(ffmpeg, mix_wav)

    if audio_only:
        # Audio-only mode: skip the video render, save just the dubbed audio
        # (loudness-matched .wav) to drop into CapCut over the original clip.
        out_audio = os.path.splitext(out_mp4)[0] + ".wav"
        print(f"__STAGE__ {lang.upper()}: exporting audio", flush=True)
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", mix_wav,
                        "-af", ln, out_audio])
        print("__PCT__ 100", flush=True)
        ok = os.path.exists(out_audio)
        print(f"  [{lang}] -> dubbed\\{os.path.basename(out_audio)}"
              + ("" if ok else "   (FAILED)"))
        if ok:
            # Machine-readable: lets the GUI pair this dub with its source for A/B preview.
            print(f"__OUT__\t{os.path.abspath(video)}\t{os.path.abspath(out_audio)}", flush=True)
            write_project(os.path.splitext(out_audio)[0] + ".dubproj.json", base=base,
                          source=video, output=out_audio, audio_only=True,
                          src_lang=source_lang, lang=lang, segs=segs, texts=texts,
                          spk_labels=spk_labels, seg_voices=seg_voices,
                          cast_roles=cast_roles, cast_genders=cast_genders,
                          preset=preset, rights_mode=rights_mode,
                          story_bible=story_bible)
        return

    print("__PCT__ 88", flush=True)
    print(f"__STAGE__ {lang.upper()}: rendering video", flush=True)
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-i", mix_wav]
    vchain = ""
    label = "[0:v]"
    sub_rect, rect_user = _parse_rect(cover_region, subs_band)
    if cover_subs:
        x, y, w, h = sub_rect
        # Heavy blur (sigma 40, multi-step) so the original burned-in subtitle is
        # unreadable, not just softened.
        vchain += (f"{label}split=2[cbase][cb];"
                   f"[cb]crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},gblur=sigma=40:steps=3[cbb];"
                   f"[cbase][cbb]overlay=W*{x}:H*{y}[cov];")
        label = "[cov]"
    if fb_vertical:
        vchain += (f"{label}split=2[fbg][ffg];"
                   f"[fbg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=22:2[fbg2];"
                   f"[ffg]scale=1080:1920:force_original_aspect_ratio=decrease[ffg2];"
                   f"[fbg2][ffg2]overlay=(W-w)/2:(H-h)/2,setsar=1[fbo];")
        label = "[fbo]"
    burn_filter = ""
    if burn_subs:
        # Burn translated subtitles as ASS, positioned over the original subtitle
        # area (the cover box) at ~the original wording size. cwd=work dodges
        # Windows path-escaping issues with the drive colon / spaces.
        out_w, out_h = (1080, 1920) if fb_vertical else ffprobe_dims(video, ffmpeg)
        out_w -= out_w % 2
        out_h -= out_h % 2
        if rect_user:
            place_rect, size_from = sub_rect, True
        elif cover_subs:
            place_rect, size_from = sub_rect, False
        else:
            place_rect, size_from = None, False
        write_ass(os.path.join(work, "_burn.ass"), segs, texts, out_w, out_h,
                  place_rect, size_from)
        burn_filter = ",subtitles=_burn.ass"
    if vchain or burn_subs:
        # force even width/height - libx264 rejects odd dimensions and writes a 0-byte file
        vchain += f"{label}scale=trunc(iw/2)*2:trunc(ih/2)*2{burn_filter}[vout];"
        fc = vchain + f"[1:a]{ln}[outa]"
        cmd += ["-filter_complex", fc, "-map", "[vout]", "-map", "[outa]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", out_mp4]
        subprocess.run(cmd, cwd=work)
    else:
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                "-af", ln, "-shortest", out_mp4]
        subprocess.run(cmd)
    print("__PCT__ 100", flush=True)
    ok = os.path.exists(out_mp4)
    print(f"  [{lang}] -> dubbed\\{base}_{lang}.mp4" + ("" if ok else "   (FAILED)"))
    if ok:
        # Machine-readable: lets the GUI pair this dub with its source for A/B preview.
        print(f"__OUT__\t{os.path.abspath(video)}\t{os.path.abspath(out_mp4)}", flush=True)
        write_project(os.path.splitext(out_mp4)[0] + ".dubproj.json", base=base,
                      source=video, output=out_mp4, audio_only=False,
                      src_lang=source_lang, lang=lang, segs=segs, texts=texts,
                      spk_labels=spk_labels, seg_voices=seg_voices,
                      cast_roles=cast_roles, cast_genders=cast_genders,
                      preset=preset, rights_mode=rights_mode,
                      story_bible=story_bible)


# ---------------------------------------------------------------------------
# Per-video analysis cache. Transcription (Whisper), the original-voice
# pitch/loudness stats, and speaker diarization are all INDEPENDENT of the
# target language - only translation + TTS + mux change per language. Caching
# them in a JSON sidecar lets a later run dub the SAME video into a DIFFERENT
# language without re-running Whisper/diarization (the slow steps). Demucs output
# is already reused on disk by ensure_bed(). The cache is keyed by the video's
# size+mtime and the Whisper model size, so editing the video or switching to a
# bigger model transparently invalidates it and forces a fresh transcription.
# ---------------------------------------------------------------------------
_CACHE_VERSION = 4        # bumped: Facebook Reels preset analysis/cache metadata


def _analysis_cache_path(work: str, base: str) -> str:
    d = os.path.join(work, "analysis")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{base}.json")


def _video_sig(video: str) -> dict:
    """Cheap fingerprint of the source video so an edited/replaced file with the
    same name doesn't reuse a stale transcription."""
    try:
        st = os.stat(video)
        return {"size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        return {}


def load_analysis(work: str, base: str, video: str, model_size: str,
                  analysis_key: str = "default") -> dict | None:
    """Return the cached analysis for this video, or None when absent/stale."""
    p = _analysis_cache_path(work, base)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if data.get("version") != _CACHE_VERSION:
        return None
    if data.get("sig") != _video_sig(video):
        return None
    if data.get("model") != model_size:        # a different model -> re-transcribe
        return None
    if data.get("analysis_key", "default") != analysis_key:
        return None
    if not data.get("segs"):
        return None
    return data


def save_analysis(work: str, base: str, video: str, model_size: str, total_ms: int,
                  src_lang: str, segs: list[dict], orig_stats: list[dict] | None,
                  median_dbfs: float | None, spk_labels: list[int] | None,
                  analysis_key: str = "default", alt_meta: dict | None = None,
                  rights_mode: str = "", cast_roles: list[str] | None = None,
                  cast_genders: list[str] | None = None) -> None:
    """Persist the language-independent analysis next to the work files. Best
    effort - a write failure never aborts the dub."""
    data = {
        "version": _CACHE_VERSION,
        "sig": _video_sig(video),
        "model": model_size,
        "analysis_key": analysis_key,
        "total_ms": total_ms,
        "src_lang": src_lang,
        "segs": segs,
        "alt_meta": alt_meta or {},
        "orig_stats": orig_stats,
        "median_dbfs": median_dbfs,
        "spk_labels": spk_labels,
        "cast_roles": cast_roles,
        "cast_genders": cast_genders,
        "rights_mode": rights_mode,
    }
    try:
        with open(_analysis_cache_path(work, base), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"  (could not cache analysis: {str(e)[:50]})", flush=True)


def normalize_speaker_override(raw) -> dict:
    """Structured job contract for caller-supplied speaker metadata.

    Example:
      {"enforce_single_speaker": true,
       "primary_speaker_id": "Speaker_1",
       "diarization_sensitivity": 0.0}

    `enforce_single_speaker` is authoritative: downstream diarization/AI casting
    is collapsed to one identity, so a noisy single-speaker clip cannot turn into
    multiple cloned voices. `diarization_sensitivity` is optional middleware for
    multi-speaker mode: 0.0 is most conservative, 1.0 is most eager to split.
    """
    if not isinstance(raw, dict):
        raw = {}
    try:
        sens = float(raw.get("diarization_sensitivity", 0.5))
    except Exception:
        sens = 0.5
    sens = _clamp(sens, 0.0, 1.0)
    primary = str(raw.get("primary_speaker_id") or "Speaker_1").strip() or "Speaker_1"
    return {
        "enforce_single_speaker": bool(raw.get("enforce_single_speaker")),
        "primary_speaker_id": primary,
        "diarization_sensitivity": sens,
    }


def diarize_once(vocals: str | None, segs: list[dict],
                 diarization_sensitivity: float = 0.5) -> list[int] | None:
    """Run speaker diarization once for the video (language-independent). Returns
    per-segment speaker labels, or None when there's one speaker / it's
    unavailable. Logged once here so make_dub() can reuse the result per language."""
    if not vocals:
        return None
    labels = None
    try:
        import speakers
        if speakers.available():
            # Sensitivity maps to the clustering split threshold. Low sensitivity
            # requires a much stronger cluster separation before creating a second
            # speaker, which protects narrator/recap videos from false splits.
            min_silhouette = 0.32 - (0.20 * _clamp(diarization_sensitivity, 0.0, 1.0))
            labels = speakers.diarize(vocals, segs, min_silhouette=min_silhouette)
    except Exception as e:
        print(f"  speaker split unavailable ({str(e)[:50]}); using one voice.", flush=True)
    if labels and len(set(labels)) > 1:
        print(f"  {len(set(labels))} speakers detected", flush=True)
        return labels
    return None        # only one speaker - nothing to split


def run_one(model, video, out_dir, ffmpeg, work, langs, keepmusic, cover_subs,
            subs_band, cover_region, fb_vertical, make_srt, naming, burn_subs=False,
            tone="natural", audio_only=False, clone=False, api_key="",
            gender_mode=False, scene_fx=False, speakers_mode=False,
            model_size="medium", reuse_analysis=True, texts_override=None,
            length_fit=False, preset: str = "", rights_mode: str = "",
            speaker_override: dict | None = None,
            multimodal: bool = False, multimodal_vision: bool = True,
            multimodal_fps: float = 1.0):
    video = os.path.abspath(video)
    base = os.path.splitext(os.path.basename(video))[0]
    speaker_override = normalize_speaker_override(speaker_override)
    facebook_reels = preset == "facebook_reels"
    if facebook_reels:
        fb_vertical = True
        burn_subs = True
        make_srt = True
        cover_subs = True
        length_fit = False
        speakers_mode = True
        gender_mode = True
        clone = False
        rights_mode = rights_mode or "owned_or_licensed"
        print("  preset: Facebook Reels Auto (owned/licensed source)", flush=True)
    if speaker_override["enforce_single_speaker"]:
        speakers_mode = False
        gender_mode = False
        print(f"  speaker override: enforcing one voice "
              f"({speaker_override['primary_speaker_id']})", flush=True)
    if multimodal and not api_key:
        print("  multimodal context: DeepSeek key missing; using local timing anchors only", flush=True)
    print(f"> {base}: dubbing -> {', '.join(langs)}", flush=True)
    bed, vocals = None, None
    # Gender mode judges male/female from the speaker's pitch, which is only
    # reliable on the isolated voice - so it needs Demucs too, not just the
    # music-keeping/cloning paths. (bed is dropped below unless keepmusic.)
    if keepmusic or clone or scene_fx or gender_mode or speakers_mode:
        bed, vocals = ensure_bed(video, base, work, ffmpeg)
    if not keepmusic:
        bed = None
    if clone and speakers_mode:
        print("  [clone] disabled for character voices; using separate cast voices.", flush=True)
        clone = False
    if clone:
        import voiceclone
        if not voiceclone.available():
            print("  [clone] OpenVoice/torch not ready; see SETUP_VOICECLONE.md. Using normal voice.", flush=True)
            clone = False
        elif not vocals:
            print("  [clone] couldn't isolate the original voice; using normal voice.", flush=True)
            clone = False
    total_ms = int(ffprobe_duration(video, ffmpeg) * 1000)

    # Reuse the language-independent analysis (transcription / voice stats /
    # speaker split) from a previous run on this same video, so dubbing it into a
    # NEW language skips Whisper + diarization and goes straight to translate+TTS.
    analysis_key = ("facebook_reels_v2_story" if facebook_reels else "default")
    if multimodal:
        analysis_key += "_mm"
    cached = load_analysis(work, base, video, model_size, analysis_key) if reuse_analysis else None
    if cached:
        segs = cached["segs"]
        src_lang = cached["src_lang"]
        total_ms = cached.get("total_ms", total_ms)
        orig_stats = cached.get("orig_stats")
        median_dbfs = cached.get("median_dbfs")
        spk_labels = None if speaker_override["enforce_single_speaker"] else cached.get("spk_labels")
        alt_meta = cached.get("alt_meta") or {}
        print(f"  reusing cached analysis: {len(segs)} segments, source "
              f"{src_lang} (skipping transcription)", flush=True)
    else:
        segs, src_lang, alt_meta = transcribe_for_dub(
            model, video, vocals, total_ms, dual_pass=facebook_reels)
        orig_stats, median_dbfs, spk_labels = None, None, None
    if not segs:
        print("  no speech detected; skipping.", flush=True)
        return

    if multimodal:
        print("__STAGE__ Analyzing visual context", flush=True)
        try:
            aligner = MultimodalAligner(ffmpeg, work, api_key=api_key,
                                        frame_sample_rate=multimodal_fps)
            mm_context = aligner.analyze(
                video, segs, total_ms=total_ms,
                cache_key=f"{base}_{analysis_key}",
                use_vision=multimodal_vision,
            )
            apply_multimodal_context(segs, mm_context)
            alt_meta["multimodal"] = mm_context.to_dict()
            if mm_context.summary:
                print(f"  visual context: {mm_context.summary[:120]}", flush=True)
            print(f"  timing anchors: {len(mm_context.lip_activity)} segments", flush=True)
        except Exception as e:
            print(f"  multimodal context skipped ({str(e)[:80]})", flush=True)

    # Compute (or back-fill) the original-voice stats when this run needs them and
    # the cache didn't already have them (e.g. first run was tone=natural).
    if (tone == "original" or gender_mode) and orig_stats is None:
        orig = load_orig_audio(video, vocals, work, base, ffmpeg)
        if orig is not None:
            orig_stats, median_dbfs = compute_orig_stats(orig, segs)
        else:
            print("  tone=original: original audio unavailable; matching pace only.", flush=True)

    # Diarize once for the whole video (not per language) and back-fill the cache.
    if speaker_override["enforce_single_speaker"]:
        spk_labels = None
    elif speakers_mode and vocals and spk_labels is None:
        spk_labels = diarize_once(
            vocals, segs, speaker_override["diarization_sensitivity"])

    save_analysis(work, base, video, model_size, total_ms, src_lang, segs,
                  orig_stats, median_dbfs, spk_labels,
                  analysis_key=analysis_key, alt_meta=alt_meta,
                  rights_mode=rights_mode)

    for lang in langs:
        make_dub(lang, segs, total_ms, bed, video, out_dir, base, work, ffmpeg, src_lang,
                 cover_subs, subs_band, cover_region, fb_vertical, make_srt, naming, burn_subs,
                 tone, orig_stats, median_dbfs, audio_only, clone, vocals, api_key, gender_mode,
                 scene_fx, spk_labels if speakers_mode else None,
                 texts_override=(texts_override or {}).get(lang),
                 length_fit=length_fit,
                 cast_mode=(speakers_mode and not speaker_override["enforce_single_speaker"]),
                 script_cleanup=facebook_reels, preset=preset,
                 rights_mode=rights_mode,
                 voice_duck=facebook_reels and bool(bed))


def _setup(ffmpeg, out_dir, work):
    out_dir = os.path.abspath(out_dir)
    work = os.path.abspath(work)
    if os.path.exists(ffmpeg):
        ffmpeg = os.path.abspath(ffmpeg)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    os.environ["PATH"] = os.path.dirname(ffmpeg) + os.pathsep + os.environ.get("PATH", "")
    AudioSegment.converter = ffmpeg
    return ffmpeg, out_dir, work


def main() -> None:
    # Batch mode: one process, model loaded once, many videos.
    if len(sys.argv) > 2 and sys.argv[1] == "--jobs":
        with open(sys.argv[2], encoding="utf-8") as f:
            cfg = json.load(f)
        ffmpeg, out_dir, work = _setup(cfg["ffmpeg"], cfg["out"], cfg["work"])
        model_size = cfg.get("model", "medium")
        langs = [x.strip() for x in cfg["langs"].split(",") if x.strip() in VOICES]
        # GUI voice picker: each ticked language carries the chosen edge-tts voice.
        # Apply it (and drop Piper for that language) so the picked voice is what plays.
        for lg, vid in (cfg.get("voices") or {}).items():
            if vid:
                VOICES[lg] = vid
                PIPER_VOICES.pop(lg, None)
        videos = cfg["videos"]
        tone = cfg.get("tone", "original")
        preset = cfg.get("preset", "")
        rights_mode = cfg.get("rights_mode", "")
        if isinstance(cfg.get("_learning_applied"), dict):
            learned = cfg["_learning_applied"]
            print(f"  learning optimizer: applied {learned.get('error_type', 'memory rule')} "
                  f"{learned.get('speaker_override', '')}", flush=True)
        model = WhisperModel(cfg.get("model", "medium"), device="cpu", compute_type="int8")
        for i, item in enumerate(videos, 1):
            if isinstance(item, str):
                video, region = item, cfg.get("region", "")
            else:
                video, region = item["path"], item.get("region", "")
            print(f"__FILE__ {i} {len(videos)} {os.path.basename(video)}", flush=True)
            try:
                run_one(model, video, out_dir, ffmpeg, work, langs,
                        cfg.get("keepmusic", False), cfg.get("cover", True),
                        cfg.get("band", 18), region, cfg.get("fb", False),
                        cfg.get("srt", False), cfg.get("naming", "source"),
                        cfg.get("burn", False), tone, cfg.get("audioonly", False),
                        cfg.get("clone", False), cfg.get("deepseek_key", ""),
                        cfg.get("gender", False), cfg.get("scenefx", False),
                        cfg.get("speakers", False), model_size,
                        cfg.get("reuse_analysis", True),
                        cfg.get("texts_override"),
                        cfg.get("length_fit", False),
                        cfg.get("preset", preset),
                        cfg.get("rights_mode", rights_mode),
                        cfg.get("speaker_override"),
                        cfg.get("multimodal", False),
                        cfg.get("multimodal_vision", True),
                        cfg.get("multimodal_fps", 1.0))
            except Exception as e:
                print(f"  ERROR on {os.path.basename(video)}: {e}", flush=True)
            print(f"__FILEDONE__ {i}", flush=True)
        return

    # Single-video mode (used by the .bat / folder buttons).
    video = sys.argv[1]
    ffmpeg, out_dir, work = _setup(sys.argv[3], sys.argv[2], sys.argv[4])
    langs = [x.strip() for x in sys.argv[5].split(",") if x.strip() in VOICES]
    model_size = sys.argv[6] if len(sys.argv) > 6 else "medium"
    keep_music = len(sys.argv) > 7 and sys.argv[7] == "1"
    cover_subs = len(sys.argv) > 8 and sys.argv[8] == "1"
    subs_band = int(sys.argv[9]) if len(sys.argv) > 9 else 18
    cover_region = sys.argv[10] if len(sys.argv) > 10 else ""
    fb_vertical = len(sys.argv) > 11 and sys.argv[11] == "1"
    make_srt = len(sys.argv) > 12 and sys.argv[12] == "1"
    naming = sys.argv[13] if len(sys.argv) > 13 else "source"
    burn = len(sys.argv) > 14 and sys.argv[14] == "1"
    tone = sys.argv[15] if len(sys.argv) > 15 else "natural"
    audio_only = len(sys.argv) > 16 and sys.argv[16] == "1"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    run_one(model, video, out_dir, ffmpeg, work, langs, keep_music, cover_subs,
            subs_band, cover_region, fb_vertical, make_srt, naming, burn, tone, audio_only,
            model_size=model_size)


if __name__ == "__main__":
    main()
