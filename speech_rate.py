#!/usr/bin/env python3
# speech_rate.py - OPTIONAL length-fit pass for dub.py.
#
# Ported/adapted from OmniVoice-Studio (backend/services/speech_rate.py) and
# decoupled from its `llm_backend`: this version takes a plain `chat(system, user)`
# callable, so dub.py can hand it the existing DeepSeek helper - NO new dependency.
#
# What it adds over dub.py's existing translation step: dub.py already tells DeepSeek
# each line's spoken-time budget ("(2.3s)") during TRANSLATION, but the model doesn't
# always obey. This module is a measurement-driven SECOND pass: it estimates each
# translated line's reading time from a per-language chars-per-second table and, only
# for the lines that still overshoot/undershoot their slot, asks the LLM to trim or
# expand JUST that line. Lines already within tolerance are left untouched and cost
# nothing. The point is to fix the text length up front so the audio fit step
# (fit_segment / MAX_FIT_TEMPO) has to time-stretch far less - less "chipmunk" speed-up.
#
# It's an add-on like xtts.py / voiceclone.py / speakers.py: if no LLM callable is
# given, fit_texts() returns the input unchanged, so the tool still runs everywhere.

from __future__ import annotations

from typing import Callable, Optional

# Per-language read-speed estimates (chars/sec at a natural pace, counting Python
# len() codepoints - not phonemes). Rough; calibrated from Pellegrino et al. 2011
# (cross-language information rate) plus informal TTS-output calibration. Codepoint
# density matters: Indic scripts encode vowel-marks as separate codepoints, inflating
# len() for the same spoken duration; CJK is logographic so fewer chars/sec.
# (Imported verbatim from OmniVoice's _RATE_CPS, trimmed to dub.py's languages + a
# sensible default for the rest.)
_RATE_CPS = {
    "en": 15.0, "de": 14.0, "fr": 15.0, "es": 15.5, "it": 15.0, "pt": 15.0,
    "nl": 14.0, "pl": 13.0, "cs": 13.0, "ru": 13.0, "tr": 12.0, "hu": 13.0,
    "ar": 12.0, "hi": 17.0,
    # CJK - logographic / mora-based, fewer chars per second.
    "ja": 10.0, "ko": 10.0, "zh": 6.0, "zh-cn": 6.0,
    # edge-tts-only languages dub.py still supports.
    "vi": 16.0, "id": 14.0, "ms": 14.0,
}
_DEFAULT_CPS = 13.0

# Tolerance window - if the predicted ratio is within this of 1.0 we accept the line
# as-is and never call the LLM for it. Wider than OmniVoice's 0.92-1.08 on the LOW
# side: dub.py can pad a short line with silence cheaply, so undershoot matters less
# than overshoot (which forces audio time-compression and drifts lip-sync).
TOL_LOW = 0.80
TOL_HIGH = 1.10

# Max LLM attempts per line before we keep the best candidate seen.
MAX_ATTEMPTS = 2


def expected_duration(text: str, lang: str = "en") -> float:
    """Rough CPS-based reading-time estimate for `text` in `lang`. Returns seconds."""
    cps = _RATE_CPS.get(lang.split("-")[0].lower(), _DEFAULT_CPS)
    return len(text or "") / max(1.0, cps)


def rate_ratio(text: str, slot_seconds: float, lang: str = "en") -> float:
    """How far the text is from filling its slot. 1.0 = perfect; >1 = too long."""
    if slot_seconds <= 0:
        return 1.0
    return expected_duration(text, lang) / slot_seconds


_TRIM_PROMPT = (
    "You are a video-dubbing writer. You will get one translated line and the exact "
    "time slot it must fit. The line is TOO LONG - trim filler, tighten phrasing, or "
    "drop less essential words while preserving the meaning. Never change character "
    "names or proper nouns. Keep it in the SAME language as the line you are given. "
    'Reply with ONLY a JSON object {"line": "..."} holding the new line.')

_EXPAND_PROMPT = (
    "You are a video-dubbing writer. You will get one translated line and the exact "
    "time slot it must fit. The line is TOO SHORT - gently flesh it out with natural "
    "wording while keeping the meaning the same. Keep it in the SAME language as the "
    'line you are given. Reply with ONLY a JSON object {"line": "..."} holding the '
    "new line.")


def _parse_line(reply: Optional[str]) -> Optional[str]:
    """Pull the rewritten line out of the model's JSON reply. Tolerant: accepts a
    bare string too. Returns None when nothing usable came back."""
    if not reply:
        return None
    import json
    try:
        obj = json.loads(reply)
        if isinstance(obj, dict):
            v = obj.get("line")
            return str(v).strip() if v is not None and str(v).strip() else None
        if isinstance(obj, str) and obj.strip():
            return obj.strip()
    except Exception:
        s = reply.strip()
        # Last resort: a model that ignored the JSON instruction and just gave text.
        if s and "{" not in s and "}" not in s and len(s) < 600:
            return s
    return None


def fit_line(text: str, slot_seconds: float, lang: str,
             chat: Callable[[str, str], Optional[str]],
             source_text: Optional[str] = None) -> str:
    """Return `text` adjusted so its estimated reading time fits `slot_seconds`,
    or the unchanged/best text if it already fits or the LLM can't improve it.
    `chat(system, user)` must return the model reply (or None on failure)."""
    if not text or not text.strip():
        return text
    r = rate_ratio(text, slot_seconds, lang)
    if TOL_LOW <= r <= TOL_HIGH:
        return text

    current = text
    best = (text, abs(r - 1.0))
    for _ in range(MAX_ATTEMPTS):
        r = rate_ratio(current, slot_seconds, lang)
        if TOL_LOW <= r <= TOL_HIGH:
            return current
        system = _TRIM_PROMPT if r > 1.0 else _EXPAND_PROMPT
        user_lines = [
            f"Slot: {slot_seconds:.2f}s",
            f"Estimated reading time: ~{expected_duration(current, lang):.2f}s (ratio {r:.2f})",
            f"Line: {current}",
        ]
        if source_text:
            user_lines.append(f"Original meaning (for reference): {source_text}")
        try:
            reply = chat(system, "\n".join(user_lines))
        except Exception:
            break
        nxt = _parse_line(reply)
        if not nxt:
            break
        current = nxt
        score = abs(rate_ratio(current, slot_seconds, lang) - 1.0)
        if score < best[1]:
            best = (current, score)
    return best[0]


def fit_texts(texts: list[str], slots_seconds: list[float], lang: str,
              chat: Optional[Callable[[str, str], Optional[str]]],
              sources: Optional[list[str]] = None) -> list[str]:
    """Length-fit a whole list of translated lines against their per-line time slots.
    Returns a same-length list. A no-op (returns `texts` unchanged) when `chat` is
    None - so callers can pass the LLM only when a key is configured. Only lines
    outside the tolerance window cost an LLM call; the rest pass straight through.

    Returns (texts, n_fitted) is NOT used - we keep the simple list contract so the
    call site stays a drop-in over the translated `texts`. The count is logged by the
    caller if it wants it via fit_texts_counted()."""
    out, _ = fit_texts_counted(texts, slots_seconds, lang, chat, sources)
    return out


def fit_texts_counted(texts: list[str], slots_seconds: list[float], lang: str,
                      chat: Optional[Callable[[str, str], Optional[str]]],
                      sources: Optional[list[str]] = None
                      ) -> tuple[list[str], int]:
    """Like fit_texts but also returns how many lines were actually changed, so the
    caller can log "fitted N/M lines"."""
    if chat is None or not texts:
        return list(texts), 0
    n = len(texts)
    src = sources if (sources and len(sources) == n) else [None] * n
    fitted = list(texts)
    changed = 0
    for i in range(n):
        slot = slots_seconds[i] if i < len(slots_seconds) else 0.0
        new = fit_line(texts[i], slot, lang, chat, src[i])
        if new != texts[i]:
            changed += 1
        fitted[i] = new
    return fitted, changed
