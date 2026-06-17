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
#   model  = faster-whisper size: tiny|base|small|medium  (default: small)

import sys
import os
import re
import time
import asyncio
import subprocess
import importlib.util

# Quieter first-run model download (these are harmless Windows/HF notices).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


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

from faster_whisper import WhisperModel          # noqa: E402
from deep_translator import GoogleTranslator, MyMemoryTranslator  # noqa: E402
import edge_tts                                    # noqa: E402
import warnings                                     # noqa: E402
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")
from pydub import AudioSegment                     # noqa: E402
try:
    import numpy as np                              # noqa: E402  (ships with faster-whisper)
except Exception:
    np = None

# Natural female neural voices. Change a value to a *Neural male voice if you prefer.
VOICES = {
    "vi": "vi-VN-HoaiMyNeural",   # Vietnamese (female)
    "id": "id-ID-GadisNeural",    # Indonesian (female)
    "ms": "ms-MY-YasminNeural",   # Malay (female)
    "es": "es-MX-DaliaNeural",    # Spanish - Latin America / Mexico (female)
    "en": "en-US-AriaNeural",     # English (US, female)
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
    "es": {"M": "es-MX-JorgeNeural", "F": "es-MX-DaliaNeural"},
    "en": {"M": "en-US-GuyNeural", "F": "en-US-AriaNeural"},
}

# Hard ceiling on how much a long line (translations often run longer than the
# source) may be sped up to fit its slot. 1.5x stays intelligible; beyond that it
# sounds chipmunky, so instead we let the timeline drift and recover at the next pause.
MAX_FIT_TEMPO = 1.5


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


def write_ass(path: str, segs: list[dict], texts: list[str], frame_w: int,
              frame_h: int, rect: list[float] | None, size_from_rect: bool) -> None:
    """Write burned subtitles as ASS, positioned over the original subtitle area
    (rect) at roughly the original wording's size. rect=None -> bottom band."""
    if rect and len(rect) == 4:
        x, y, w, h = rect
    else:
        x, y, w, h = 0.0, 0.80, 1.0, 0.16
    if size_from_rect:
        fs = int(_clamp(round(h * frame_h * 0.72), 22, round(frame_h * 0.12)))
    else:
        fs = max(22, round(frame_h * 0.045))
    pad = round(0.015 * frame_h)
    if (y + h / 2.0) <= 0.5:
        align, mv = 8, max(pad, round(y * frame_h))                 # top-center
    else:
        align, mv = 2, max(pad, round((1.0 - (y + h)) * frame_h))   # bottom-center
    ml = max(0, round(x * frame_w))
    mr = max(0, round((1.0 - (x + w)) * frame_w))
    outline = max(2, round(fs * 0.07))
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {frame_w}\n"
        f"PlayResY: {frame_h}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Arial,{fs},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        f"0,0,0,0,100,100,0,0,1,{outline},0,{align},{ml},{mr},{mv},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for s, t in zip(segs, texts):
            t = (t or "").strip()
            if not t:
                continue
            t = t.replace("\\", "/").replace("{", "(").replace("}", ")")
            t = t.replace("\r\n", "\\N").replace("\n", "\\N").replace("\r", "\\N")
            f.write(f"Dialogue: 0,{ass_ts(s['start'])},{ass_ts(s['end'])},"
                    f"Default,,0,0,0,,{t}\n")


def ensure_bed(video: str, base: str, work: str, ffmpeg: str) -> tuple[str | None, str | None]:
    """Return (music_bed, clean_vocals) paths, reusing Demucs output if present."""
    bed = os.path.join(work, "htdemucs", base, "no_vocals.wav")
    vocals = os.path.join(work, "htdemucs", base, "vocals.wav")
    if os.path.exists(bed):
        return bed, (vocals if os.path.exists(vocals) else None)

    stem = os.path.join(work, base + ".wav")
    run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-i", video, "-vn", "-ac", "2", "-ar", "44100", stem])
    print("  separating voice/music with Demucs (first run downloads ~80MB model)...")
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems", "vocals", "-o", work, stem])
    if os.path.exists(bed):
        return bed, (vocals if os.path.exists(vocals) else None)
    print("  Demucs unavailable; original music will be dropped (voice-only dub).")
    return None, None


def transcribe(model, audio_path: str, total_ms: int = 0) -> tuple[list[dict], str]:
    print("__STAGE__ Transcribing audio", flush=True)
    # Douyin clips are music-heavy; without these, Whisper loops and re-emits the last
    # line as hundreds of duplicate Chinese segments (the "repeating" bug). Disabling
    # previous-text conditioning + an n-gram repeat guard stops the loop.
    segments, info = model.transcribe(
        audio_path, vad_filter=True,
        condition_on_previous_text=False, no_repeat_ngram_size=3)
    out: list[dict] = []
    for s in segments:
        text = (s.text or "").strip()
        if text:
            out.append({"start": s.start, "end": s.end, "text": text})
        if total_ms > 0:
            print(f"__PCT__ {min(99, int(s.end * 1000 / total_ms * 100))}", flush=True)
    print(f"  source language: {info.language}   segments: {len(out)}", flush=True)
    return out, info.language


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
            for attempt in range(2):
                try:
                    await edge_tts.Communicate(text, v).save(path)
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
        src = os.path.join(work, base + "_orig16k.wav")
        if not os.path.exists(src):
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
    tempo = _clamp(length / target_ms, 0.7, MAX_FIT_TEMPO)
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
_MM_TARGET = {"vi": "vi-VN", "id": "id-ID", "ms": "ms-MY", "es": "es-ES", "en": "en-US"}
_MM_SOURCE = {"zh": "zh-CN", "ko": "ko-KR", "ja": "ja-JP", "en": "en-GB",
              "ms": "ms-MY", "th": "th-TH", "vi": "vi-VN", "id": "id-ID", "es": "es-ES"}


def _chunk(lines: list[str], max_chars: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for ln in lines:
        if cur and size + len(ln) + 1 > max_chars:
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
    blank it so it is skipped rather than voiced as gibberish."""
    if lang.startswith("zh"):
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
               "es": "Latin American Spanish", "en": "English"}


def _deepseek_key(explicit: str = "") -> str:
    return (explicit or os.environ.get("DEEPSEEK_API_KEY", "")).strip()


def _deepseek_chat(messages: list[dict], api_key: str) -> str | None:
    """One OpenAI-compatible chat call to DeepSeek. Returns the reply text, or
    None on any failure (network, auth, bad JSON) so the caller can fall back."""
    import json as _json
    import urllib.request
    body = _json.dumps({"model": DEEPSEEK_MODEL, "messages": messages,
                        "temperature": 1.3, "stream": False,
                        "response_format": {"type": "json_object"}}).encode("utf-8")
    req = urllib.request.Request(
        DEEPSEEK_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = _json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def _deepseek_translate(lines: list[str], lang: str, api_key: str,
                        durations: list[float] | None = None) -> list[str] | None:
    """Translate every line with DeepSeek. Returns a same-length list, or None if
    a call fails or the model doesn't return exactly one translation per line.
    durations: per-line spoken-time budget (seconds). When given, each line is
    tagged with its budget and the model is told to keep the translation short
    enough to say in that time - this is what keeps the dub from drifting out of
    sync when a translation would otherwise run longer than the original line."""
    import json as _json
    target = _LANG_NAMES.get(lang, lang)
    out: list[str] = []
    pos = 0
    for chunk in _chunk(lines, 3500):
        if durations is not None:
            numbered = "\n".join(f"{i + 1}. ({durations[pos + i]:.1f}s) {ln}"
                                 for i, ln in enumerate(chunk))
            budget_rule = (
                "Each line is prefixed with its spoken-time budget in seconds, "
                "like (2.3s). Your translation MUST be short enough to say "
                "naturally within that budget - tighten phrasing and cut filler "
                "rather than overrun, while keeping the core meaning. ")
        else:
            numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(chunk))
            budget_rule = ("Keep each line about as short as the source so it "
                           "fits the original speaking time. ")
        sys_msg = (
            f"You are a professional video-dubbing translator. Translate each "
            f"numbered line into natural, spoken {target}, the way a voice actor "
            f"would say it. {budget_rule}Translate fully - never keep or "
            f"transliterate the source language. Reply with ONLY a JSON object "
            f'{{"lines": [...]}} holding exactly {len(chunk)} strings, in order, '
            f"one per input line.")
        content = _deepseek_chat(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": numbered}], api_key)
        if not content:
            return None
        try:
            parts = _json.loads(content).get("lines")
        except Exception:
            parts = None
        if not isinstance(parts, list) or len(parts) != len(chunk):
            return None
        out.extend("" if p is None else str(p) for p in parts)
        pos += len(chunk)
    return out


def translate_segments(lines: list[str], lang: str, source_lang: str = "auto",
                       api_key: str = "",
                       durations: list[float] | None = None) -> list[str]:
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
        res = _deepseek_translate(lines, lang, key, durations)
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


def make_dub(lang: str, segs: list[dict], total_ms: int, bed: str | None,
             video: str, out_dir: str, base: str, work: str, ffmpeg: str,
             source_lang: str = "auto", cover_subs: bool = False, subs_band: int = 18,
             cover_region: str = "", fb_vertical: bool = False,
             make_srt: bool = False, naming: str = "source", burn_subs: bool = False,
             tone: str = "natural", orig_stats: list[dict] | None = None,
             median_dbfs: float | None = None, audio_only: bool = False,
             clone: bool = False, vocals: str | None = None,
             api_key: str = "", gender_mode: bool = False,
             scene_fx: bool = False) -> None:
    if naming != "firstline":
        out_mp4 = os.path.join(out_dir, f"{base}_{lang}.mp4")
        if os.path.exists(out_mp4):
            print(f"  [{lang}] already done; skipping")
            return

    print(f"__STAGE__ {lang.upper()}: translating", flush=True)
    # Budget each translation to the original line's spoken time so DeepSeek keeps
    # it short enough to stay in sync (floor avoids an impossible budget on blips).
    durations = [max(0.6, s["end"] - s["start"]) for s in segs]
    texts = translate_segments([s["text"] for s in segs], lang, source_lang,
                               api_key, durations)
    print("__PCT__ 12", flush=True)

    if naming == "firstline":
        first = next((t for t in texts if t.strip()), "")
        out_mp4 = _unique_out(out_dir, _safe_name(first) or base, lang)

    if make_srt:
        write_srt(os.path.splitext(out_mp4)[0] + ".srt", segs, texts)

    seg_dir = os.path.join(work, f"tts_{base}_{lang}")
    os.makedirs(seg_dir, exist_ok=True)
    jobs = [(texts[i], os.path.join(seg_dir, f"{i:03d}")) for i in range(len(texts))]
    seg_voices = None
    genders = None
    if gender_mode and lang in GENDER_VOICES:
        genders = classify_genders(orig_stats, len(jobs))
        seg_voices = [GENDER_VOICES[lang][g] for g in genders]
        nm = sum(1 for g in genders if g == "M")
        print(f"  [{lang}] gender voices: {nm} male / {len(genders) - nm} female lines", flush=True)
    print(f"__STAGE__ {lang.upper()}: voicing", flush=True)
    # English: prefer XTTS v2 (local, natural, and clones the original speaker from
    # the vocals stem when one is present). It supersedes Piper AND the OpenVoice
    # clone step for en; when coqui-tts / the model aren't installed available() is
    # False and we fall back to the normal Piper/edge-tts path. Other languages
    # (ms/id/vi/es) are unchanged - XTTS doesn't support them.
    import xtts
    use_xtts = lang == "en" and xtts.available()
    if use_xtts:
        # Clone the original speaker ONLY when the user explicitly ticked "clone".
        # Cloning a Chinese source voice into English adds a heavy accent that wrecks
        # clarity, and `vocals` can exist for unrelated reasons (gender / scene-fx /
        # keep-music all run Demucs), so we must NOT clone by default - use XTTS's
        # clean built-in English speaker instead.
        paths = xtts.synthesize(jobs, vocals if clone else None, genders)
    else:
        paths = tts_all(jobs, lang, seg_voices)
    if clone and vocals and not use_xtts:
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
        start = max(orig_start, cursor)                       # never overlap the last line
        nxt = int(segs[i + 1]["start"] * 1000) if i + 1 < len(segs) else total_ms
        room = max(300, nxt - start)                          # time left before the next line
        fit_wav = os.path.join(seg_dir, f"{i:03d}_fit.wav")
        if use_xtts:
            # XTTS speech is already natural but runs a bit longer than the source.
            # Pitch-warping (tone=original) or hard time-compression mangles neural
            # speech into the "can't understand it" mush, so for XTTS we do neither:
            # no pitch shift, only a mild speed-up, and let the timeline drift and
            # recover at the next pause instead of cramming every line into its slot.
            seg_audio = fit_segment(path, fit_wav, room, ffmpeg, max_tempo=1.25)
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
        music = music[:total_ms] - 6
        mixed = music.overlay(voice_track)
    else:
        mixed = voice_track

    if scene_fx and vocals and os.path.exists(vocals):
        print(f"  [{lang}] keeping scene sounds (panting/shouts) from non-dialogue gaps", flush=True)
        mixed = mixed.overlay(build_scene_fx(vocals, segs, total_ms))

    mix_wav = os.path.join(work, f"{base}_{lang}_mix.wav")
    mixed.export(mix_wav, format="wav")

    if audio_only:
        # Audio-only mode: skip the video render, save just the dubbed audio
        # (loudness-matched .wav) to drop into CapCut over the original clip.
        out_audio = os.path.splitext(out_mp4)[0] + ".wav"
        print(f"__STAGE__ {lang.upper()}: exporting audio", flush=True)
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", mix_wav,
                        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", out_audio])
        print("__PCT__ 100", flush=True)
        ok = os.path.exists(out_audio)
        print(f"  [{lang}] -> dubbed\\{os.path.basename(out_audio)}"
              + ("" if ok else "   (FAILED)"))
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
        fc = vchain + "[1:a]loudnorm=I=-14:TP=-1.5:LRA=11[outa]"
        cmd += ["-filter_complex", fc, "-map", "[vout]", "-map", "[outa]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", out_mp4]
        subprocess.run(cmd, cwd=work)
    else:
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-shortest", out_mp4]
        subprocess.run(cmd)
    print("__PCT__ 100", flush=True)
    ok = os.path.exists(out_mp4)
    print(f"  [{lang}] -> dubbed\\{base}_{lang}.mp4" + ("" if ok else "   (FAILED)"))


def run_one(model, video, out_dir, ffmpeg, work, langs, keepmusic, cover_subs,
            subs_band, cover_region, fb_vertical, make_srt, naming, burn_subs=False,
            tone="natural", audio_only=False, clone=False, api_key="",
            gender_mode=False, scene_fx=False):
    video = os.path.abspath(video)
    base = os.path.splitext(os.path.basename(video))[0]
    print(f"> {base}: dubbing -> {', '.join(langs)}", flush=True)
    bed, vocals = None, None
    # Gender mode judges male/female from the speaker's pitch, which is only
    # reliable on the isolated voice - so it needs Demucs too, not just the
    # music-keeping/cloning paths. (bed is dropped below unless keepmusic.)
    if keepmusic or clone or scene_fx or gender_mode:
        bed, vocals = ensure_bed(video, base, work, ffmpeg)
    if not keepmusic:
        bed = None
    if clone:
        import voiceclone
        if not voiceclone.available():
            print("  [clone] OpenVoice/torch not ready; see SETUP_VOICECLONE.md. Using normal voice.", flush=True)
            clone = False
        elif not vocals:
            print("  [clone] couldn't isolate the original voice; using normal voice.", flush=True)
            clone = False
    total_ms = int(ffprobe_duration(video, ffmpeg) * 1000)
    segs, src_lang = transcribe(model, vocals if vocals else video, total_ms)
    if not segs:
        print("  no speech detected; skipping.", flush=True)
        return
    orig_stats, median_dbfs = None, None
    if tone == "original" or gender_mode:
        orig = load_orig_audio(video, vocals, work, base, ffmpeg)
        if orig is not None:
            orig_stats, median_dbfs = compute_orig_stats(orig, segs)
        else:
            print("  tone=original: original audio unavailable; matching pace only.", flush=True)
    for lang in langs:
        make_dub(lang, segs, total_ms, bed, video, out_dir, base, work, ffmpeg, src_lang,
                 cover_subs, subs_band, cover_region, fb_vertical, make_srt, naming, burn_subs,
                 tone, orig_stats, median_dbfs, audio_only, clone, vocals, api_key, gender_mode,
                 scene_fx)


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
        import json
        with open(sys.argv[2], encoding="utf-8") as f:
            cfg = json.load(f)
        ffmpeg, out_dir, work = _setup(cfg["ffmpeg"], cfg["out"], cfg["work"])
        langs = [x.strip() for x in cfg["langs"].split(",") if x.strip() in VOICES]
        # GUI voice picker: each ticked language carries the chosen edge-tts voice.
        # Apply it (and drop Piper for that language) so the picked voice is what plays.
        for lg, vid in (cfg.get("voices") or {}).items():
            if vid:
                VOICES[lg] = vid
                PIPER_VOICES.pop(lg, None)
        videos = cfg["videos"]
        tone = cfg.get("tone", "original")
        model = WhisperModel(cfg.get("model", "small"), device="cpu", compute_type="int8")
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
                        cfg.get("gender", False), cfg.get("scenefx", False))
            except Exception as e:
                print(f"  ERROR on {os.path.basename(video)}: {e}", flush=True)
            print(f"__FILEDONE__ {i}", flush=True)
        return

    # Single-video mode (used by the .bat / folder buttons).
    video = sys.argv[1]
    ffmpeg, out_dir, work = _setup(sys.argv[3], sys.argv[2], sys.argv[4])
    langs = [x.strip() for x in sys.argv[5].split(",") if x.strip() in VOICES]
    model_size = sys.argv[6] if len(sys.argv) > 6 else "small"
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
            subs_band, cover_region, fb_vertical, make_srt, naming, burn, tone, audio_only)


if __name__ == "__main__":
    main()
