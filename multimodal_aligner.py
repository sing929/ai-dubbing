#!/usr/bin/env python3
"""Multimodal video context and timing anchors for AI dubbing."""

from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_VISION_MODEL = os.environ.get("DEEPSEEK_VISION_MODEL", "deepseek-vl2")


@dataclass(slots=True)
class FrameObservation:
    timestamp: float
    path: str
    trigger: str = "periodic"
    setting: str = ""
    speaker_expression: str = ""
    visual_context: str = ""
    ambiguous_terms: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class VisionSceneModifier:
    start: float
    end: float
    vibe: str = ""
    pacing: str = ""
    visual_scene_type: str = ""
    instruction: str = ""
    confidence: float = 0.5

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class LipActivityAnchor:
    segment_index: int
    start: float
    end: float
    mouth_open_start: float
    mouth_closed_end: float
    confidence: float
    method: str = "transcript"

    @property
    def spoken_budget(self) -> float:
        return max(0.35, self.mouth_closed_end - self.mouth_open_start)


@dataclass(slots=True)
class MultimodalContext:
    video_id: str
    created_at: float
    frame_sample_rate: float
    observations: list[FrameObservation]
    lip_activity: list[LipActivityAnchor]
    scene_modifiers: list[VisionSceneModifier] = field(default_factory=list)
    summary: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultimodalContext":
        return cls(
            video_id=str(data.get("video_id") or ""),
            created_at=float(data.get("created_at") or time.time()),
            frame_sample_rate=float(data.get("frame_sample_rate") or 1.0),
            observations=[
                FrameObservation(**item)
                for item in data.get("observations", [])
                if isinstance(item, dict)
            ],
            lip_activity=[
                LipActivityAnchor(**item)
                for item in data.get("lip_activity", [])
                if isinstance(item, dict)
            ],
            scene_modifiers=[
                VisionSceneModifier(**item)
                for item in data.get("scene_modifiers", [])
                if isinstance(item, dict)
            ],
            summary=str(data.get("summary") or ""),
            warnings=[str(x) for x in data.get("warnings", [])],
        )


@dataclass(slots=True)
class AlignmentDecision:
    segment_index: int
    text: str
    visual_start: float
    visual_end: float
    visual_budget: float
    estimated_duration: float
    cps: float
    max_cps: float
    stretch_ratio: float
    action: str


class DeepSeekVisionClient:
    """OpenAI-compatible client for DeepSeek-VL style multimodal chat."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEEPSEEK_VISION_MODEL,
        url: str = DEEPSEEK_URL,
        timeout_seconds: int = 180,
    ) -> None:
        self.api_key = (api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        self.model = model
        self.url = url
        self.timeout_seconds = timeout_seconds

    def analyze_frames(
        self,
        frames: list[Path],
        transcript_excerpt: str,
        timestamps: list[float],
        translated_script: str = "",
    ) -> dict[str, Any] | None:
        if not self.api_key or not frames:
            return None
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                "Analyze these sparse, cost-trimmed keyframes for a video dubbing "
                "alignment engine. Return ONLY JSON with summary, observations, and "
                "scene_modifiers. Each observation should include timestamp, setting, "
                "speaker_expression, visual_context, ambiguous_terms, and trigger. "
                "Each scene_modifiers item must include vibe, pacing, visual_scene_type, "
                "visual_boundaries as [start_seconds, end_seconds], instruction, and "
                "confidence. Use these as SOFT constraints; downstream code enforces "
                "timing deterministically.\n\n"
                f"Transcript excerpt:\n{transcript_excerpt[:5000]}\n\n"
                f"Translated script excerpt:\n{translated_script[:5000]}"
            ),
        }]
        for frame in frames[:8]:
            try:
                b64 = base64.b64encode(frame.read_bytes()).decode("ascii")
            except OSError:
                continue
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content_text = payload["choices"][0]["message"]["content"]
            data = json.loads(content_text)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"DeepSeek vision request failed: {exc}") from exc
        if not isinstance(data, dict):
            return None
        for idx, item in enumerate(data.get("observations") or []):
            if isinstance(item, dict) and "timestamp" not in item and idx < len(timestamps):
                item["timestamp"] = timestamps[idx]
        return data


class MultimodalAligner:
    """Samples frames, asks vision for context, and derives timing budgets."""

    def __init__(
        self,
        ffmpeg: str,
        work_dir: str | Path,
        api_key: str = "",
        frame_sample_rate: float = 1.0,
    ) -> None:
        self.ffmpeg = ffmpeg
        self.work_dir = Path(work_dir)
        self.api_key = api_key
        self.frame_sample_rate = max(0.2, float(frame_sample_rate or 1.0))
        self.analysis_dir = self.work_dir / "analysis" / "multimodal"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        video: str,
        segs: list[dict[str, Any]],
        *,
        total_ms: int = 0,
        cache_key: str = "",
        use_vision: bool = True,
        translated_script: str = "",
    ) -> MultimodalContext:
        video_id = cache_key or Path(video).stem
        cache_path = self.analysis_dir / f"{_safe_stem(video_id)}_multimodal.json"
        if cache_path.exists():
            try:
                return MultimodalContext.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        warnings: list[str] = []
        duration = max(total_ms / 1000.0, _last_segment_end(segs))
        frames, timestamps, triggers = self._sample_frames(video, video_id, duration, segs)
        observations: list[FrameObservation] = [
            FrameObservation(timestamp=ts, path=str(path), trigger=triggers[idx] if idx < len(triggers) else "unknown")
            for idx, (path, ts) in enumerate(zip(frames, timestamps))
        ]
        scene_modifiers: list[VisionSceneModifier] = []
        summary = ""
        if use_vision and self.api_key and frames:
            try:
                data = DeepSeekVisionClient(self.api_key).analyze_frames(
                    frames,
                    transcript_excerpt=_transcript_excerpt(segs),
                    timestamps=timestamps,
                    translated_script=translated_script,
                )
                if data:
                    summary = str(data.get("summary") or "")
                    observations = _observations_from_vision(data, frames, timestamps, triggers)
                    scene_modifiers = _scene_modifiers_from_vision(data, duration)
            except Exception as exc:
                warnings.append(str(exc)[:300])
        elif not self.api_key:
            warnings.append("DeepSeek vision skipped: no API key.")
        lip_activity = self._lip_anchors(segs, duration)
        context = MultimodalContext(
            video_id=video_id,
            created_at=time.time(),
            frame_sample_rate=self.frame_sample_rate,
            observations=observations,
            lip_activity=lip_activity,
            scene_modifiers=scene_modifiers,
            summary=summary,
            warnings=warnings,
        )
        try:
            cache_path.write_text(json.dumps(context.to_dict(), ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return context

    def _sample_frames(
        self,
        video: str,
        video_id: str,
        duration: float,
        segs: list[dict[str, Any]],
    ) -> tuple[list[Path], list[float], list[str]]:
        frame_dir = self.analysis_dir / f"{_safe_stem(video_id)}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        keyframes = self._keyframe_plan(video, duration, segs)
        timestamps = [ts for ts, _trigger in keyframes]
        triggers = [trigger for _ts, trigger in keyframes]
        out: list[Path] = []
        kept_ts: list[float] = []
        kept_triggers: list[str] = []
        for idx, ts in enumerate(timestamps):
            path = frame_dir / f"frame_{idx:03d}_{int(ts * 1000):08d}.jpg"
            if not path.exists():
                cmd = [
                    self.ffmpeg,
                    "-y", "-ss", f"{ts:.3f}", "-i", video,
                    "-frames:v", "1",
                    "-vf", "scale=256:-2,format=gray,eq=contrast=1.35:brightness=0.03",
                    "-q:v", "8", str(path),
                ]
                try:
                    subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
                except Exception:
                    pass
            if path.exists():
                out.append(path)
                kept_ts.append(ts)
                kept_triggers.append(triggers[idx] if idx < len(triggers) else "unknown")
        return out, kept_ts, kept_triggers

    def _keyframe_plan(
        self,
        video: str,
        duration: float,
        segs: list[dict[str, Any]],
    ) -> list[tuple[float, str]]:
        events: list[tuple[float, str]] = []
        for idx, seg in enumerate(segs):
            start = round(max(0.0, float(seg.get("start", 0.0))), 3)
            if idx == 0 or _is_new_sentence_or_speaker(segs, idx):
                events.append((start, "speaker_sentence_start"))
        for ts in self._detect_scene_cuts(video, duration):
            events.append((ts, "scene_cut"))
        if not events:
            events.append((0.0, "fallback_start"))
        merged: list[tuple[float, str]] = []
        for ts, trigger in sorted(events, key=lambda item: item[0]):
            ts = round(min(max(0.0, ts), max(0.0, duration)), 3)
            if merged and abs(ts - merged[-1][0]) < 0.18:
                prev_ts, prev_trigger = merged[-1]
                merged[-1] = (prev_ts, _merge_trigger(prev_trigger, trigger))
                continue
            merged.append((ts, trigger))
        max_frames = max(4, min(24, int(math.ceil(duration / max(1.0, 4.0 / self.frame_sample_rate))) or 4))
        if len(merged) <= max_frames:
            return merged
        priority = {"scene_cut": 0, "scene_cut+speaker_sentence_start": 0, "speaker_sentence_start": 1}
        ranked = sorted(enumerate(merged), key=lambda item: (priority.get(item[1][1], 2), item[0]))
        keep_indexes = {idx for idx, _item in ranked[:max_frames]}
        return [item for idx, item in enumerate(merged) if idx in keep_indexes]

    def _detect_scene_cuts(self, video: str, duration: float) -> list[float]:
        if duration <= 0:
            return []
        fps = max(0.5, min(3.0, self.frame_sample_rate * 2.0))
        width, height = 64, 36
        cmd = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", video,
            "-vf", f"fps={fps:.3f},scale={width}:{height},format=gray",
            "-f", "rawvideo", "pipe:1",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=90)
            raw = proc.stdout
        except Exception:
            return []
        frame_size = width * height
        if len(raw) < frame_size * 2:
            return []
        try:
            import numpy as _np
        except Exception:
            return []
        cuts: list[float] = []
        prev = None
        frame_count = len(raw) // frame_size
        for idx in range(frame_count):
            frame = _np.frombuffer(raw[idx * frame_size:(idx + 1) * frame_size], dtype=_np.uint8)
            if prev is None:
                prev = frame.astype(_np.int16)
                continue
            cur = frame.astype(_np.int16)
            mad = float(_np.mean(_np.abs(cur - prev)))
            var_delta = abs(float(_np.var(cur)) - float(_np.var(prev)))
            if mad > 28.0 or (mad > 18.0 and var_delta > 180.0):
                ts = round(idx / fps, 3)
                if not cuts or ts - cuts[-1] > 0.7:
                    cuts.append(ts)
            prev = cur
        return cuts[:32]

    def _lip_anchors(self, segs: list[dict[str, Any]], duration: float) -> list[LipActivityAnchor]:
        anchors: list[LipActivityAnchor] = []
        for idx, seg in enumerate(segs):
            start = max(0.0, float(seg.get("start", 0.0)))
            end = max(start, float(seg.get("end", start)))
            next_start = duration
            if idx + 1 < len(segs):
                next_start = max(end, float(segs[idx + 1].get("start", end)))
            visual_end = min(end, max(start, next_start - 0.08))
            if visual_end - start > 1.0:
                visual_end -= 0.04
            anchors.append(LipActivityAnchor(
                segment_index=idx,
                start=start,
                end=end,
                mouth_open_start=start,
                mouth_closed_end=max(start + 0.35, visual_end),
                confidence=0.55,
                method="transcript_gap_anchor",
            ))
        return anchors


def apply_multimodal_context(segs: list[dict[str, Any]], context: MultimodalContext) -> None:
    """Attach visual context and spoken budgets to transcript segments in-place."""

    obs_by_time = sorted(context.observations, key=lambda x: x.timestamp)
    anchors = {a.segment_index: a for a in context.lip_activity}
    for idx, seg in enumerate(segs):
        anchor = anchors.get(idx)
        if anchor:
            seg["visual_start"] = round(anchor.mouth_open_start, 3)
            seg["visual_end"] = round(anchor.mouth_closed_end, 3)
            seg["visual_budget"] = round(anchor.spoken_budget, 3)
            seg["lip_activity_confidence"] = anchor.confidence
        modifier = _scene_modifier_for_segment(context.scene_modifiers, float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
        if modifier:
            seg["vision_semantics"] = {
                "vibe": modifier.vibe,
                "pacing": modifier.pacing,
                "visual_scene_type": modifier.visual_scene_type,
                "visual_boundaries": [round(modifier.start, 3), round(modifier.end, 3)],
                "instruction": modifier.instruction,
                "confidence": modifier.confidence,
            }
            if modifier.duration > 0:
                seg["scene_visual_start"] = round(modifier.start, 3)
                seg["scene_visual_end"] = round(modifier.end, 3)
        obs = _nearest_observation(obs_by_time, float(seg.get("start", 0.0)))
        if obs:
            seg["visual_context"] = {
                "timestamp": obs.timestamp,
                "trigger": obs.trigger,
                "setting": obs.setting,
                "speaker_expression": obs.speaker_expression,
                "visual_context": obs.visual_context,
                "ambiguous_terms": obs.ambiguous_terms,
            }


def enforce_alignment_constraints(
    segs: list[dict[str, Any]],
    texts: list[str],
    lang: str,
    chat: Any | None = None,
    *,
    max_tempo: float = 1.45,
    max_cps: float | None = None,
) -> tuple[list[str], list[AlignmentDecision]]:
    """Deterministically fit translated text to visual boundaries before TTS.

    DeepSeek-Vision can suggest semantic scene boundaries, but this function treats
    the numeric boundaries as hard slots. If estimated speech would require more
    than max_tempo time compression, it asks the text model for a shorter line.
    """

    out = list(texts)
    decisions: list[AlignmentDecision] = []
    lang_cps = max_cps or _natural_cps(lang) * 1.18
    for idx, seg in enumerate(segs):
        if idx >= len(out):
            break
        text = out[idx]
        start = float(seg.get("visual_start", seg.get("start", 0.0)) or 0.0)
        end = float(seg.get("visual_end", seg.get("end", start)) or start)
        semantics = seg.get("vision_semantics") if isinstance(seg.get("vision_semantics"), dict) else {}
        bounds = semantics.get("visual_boundaries") if isinstance(semantics, dict) else None
        if isinstance(bounds, list) and len(bounds) == 2:
            try:
                b0, b1 = float(bounds[0]), float(bounds[1])
                if b1 > b0 and start >= b0 - 0.05 and start < b1 + 0.05:
                    end = min(end, b1) if end > start else b1
            except Exception:
                pass
        budget = max(0.35, end - start)
        estimated = _estimated_spoken_duration(text, lang)
        cps = len(text or "") / budget if budget > 0 else 0.0
        stretch_ratio = estimated / budget if budget > 0 else 1.0
        action = "accept"
        if cps > lang_cps or stretch_ratio > max_tempo:
            action = "truncate_with_llm" if chat else "needs_truncation"
            if chat:
                shorter = _truncate_for_boundary(
                    chat=chat,
                    text=text,
                    source=str(seg.get("text") or ""),
                    lang=lang,
                    slot_seconds=budget,
                    max_words=max(3, int(5 * budget)),
                    visual_context=seg.get("visual_context") or {},
                    semantics=semantics,
                )
                if shorter:
                    out[idx] = shorter
                    text = shorter
                    estimated = _estimated_spoken_duration(text, lang)
                    cps = len(text or "") / budget if budget > 0 else 0.0
                    stretch_ratio = estimated / budget if budget > 0 else 1.0
                    action = "truncated" if stretch_ratio <= max_tempo else "truncate_requested_still_long"
        elif stretch_ratio > 1.03:
            action = "time_stretch"
        seg["alignment"] = {
            "cps": round(cps, 3),
            "max_cps": round(lang_cps, 3),
            "estimated_duration": round(estimated, 3),
            "stretch_ratio": round(stretch_ratio, 3),
            "action": action,
        }
        decisions.append(AlignmentDecision(
            segment_index=idx,
            text=text,
            visual_start=start,
            visual_end=end,
            visual_budget=budget,
            estimated_duration=estimated,
            cps=cps,
            max_cps=lang_cps,
            stretch_ratio=stretch_ratio,
            action=action,
        ))
    return out, decisions


def _observations_from_vision(
    data: dict[str, Any],
    frames: list[Path],
    timestamps: list[float],
    triggers: list[str] | None = None,
) -> list[FrameObservation]:
    observations: list[FrameObservation] = []
    raw = data.get("observations") or []
    if not isinstance(raw, list):
        raw = []
    for idx, item in enumerate(raw[:len(frames)]):
        if not isinstance(item, dict):
            continue
        terms = item.get("ambiguous_terms")
        observations.append(FrameObservation(
            timestamp=float(item.get("timestamp", timestamps[idx] if idx < len(timestamps) else 0.0)),
            path=str(frames[idx]) if idx < len(frames) else "",
            trigger=str(item.get("trigger") or (triggers[idx] if triggers and idx < len(triggers) else "vision")),
            setting=str(item.get("setting") or ""),
            speaker_expression=str(item.get("speaker_expression") or ""),
            visual_context=str(item.get("visual_context") or ""),
            ambiguous_terms=terms if isinstance(terms, dict) else {},
        ))
    if observations:
        return observations
    return [
        FrameObservation(timestamp=ts, path=str(path), trigger=triggers[idx] if triggers and idx < len(triggers) else "sampled")
        for idx, (path, ts) in enumerate(zip(frames, timestamps))
    ]


def _scene_modifiers_from_vision(data: dict[str, Any], duration: float) -> list[VisionSceneModifier]:
    raw = data.get("scene_modifiers") or data.get("semantic_context_modifiers") or []
    if not isinstance(raw, list):
        return []
    out: list[VisionSceneModifier] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        bounds = item.get("visual_boundaries") or item.get("boundaries")
        if not isinstance(bounds, list) or len(bounds) != 2:
            continue
        try:
            start = max(0.0, float(bounds[0]))
            boundary_end = float(bounds[1])
            end = max(start, min(duration if duration > 0 else boundary_end, boundary_end))
        except Exception:
            continue
        if end <= start:
            continue
        out.append(VisionSceneModifier(
            start=round(start, 3),
            end=round(end, 3),
            vibe=str(item.get("vibe") or ""),
            pacing=str(item.get("pacing") or ""),
            visual_scene_type=str(item.get("visual_scene_type") or item.get("scene_type") or ""),
            instruction=str(item.get("instruction") or item.get("modifier") or ""),
            confidence=float(item.get("confidence", 0.5) or 0.5),
        ))
    return out


def _nearest_observation(observations: list[FrameObservation], ts: float) -> FrameObservation | None:
    if not observations:
        return None
    return min(observations, key=lambda obs: abs(obs.timestamp - ts))


def _scene_modifier_for_segment(
    modifiers: list[VisionSceneModifier],
    start: float,
    end: float,
) -> VisionSceneModifier | None:
    if not modifiers:
        return None
    mid = (start + end) / 2.0
    containing = [m for m in modifiers if m.start <= mid <= m.end]
    if containing:
        return max(containing, key=lambda m: m.confidence)
    return min(modifiers, key=lambda m: abs(((m.start + m.end) / 2.0) - mid))


def _is_new_sentence_or_speaker(segs: list[dict[str, Any]], idx: int) -> bool:
    cur = segs[idx]
    prev = segs[idx - 1] if idx > 0 else {}
    if cur.get("speaker") is not None and prev.get("speaker") is not None and cur.get("speaker") != prev.get("speaker"):
        return True
    gap = float(cur.get("start", 0.0)) - float(prev.get("end", cur.get("start", 0.0)))
    if gap >= 0.25:
        return True
    prev_text = str(prev.get("text") or "").strip()
    return prev_text.endswith((".", "?", "!", "。", "？", "！"))


def _merge_trigger(left: str, right: str) -> str:
    parts = []
    for item in (left, right):
        for bit in item.split("+"):
            if bit and bit not in parts:
                parts.append(bit)
    return "+".join(parts)


def _natural_cps(lang: str) -> float:
    table = {
        "en": 15.0, "de": 14.0, "fr": 15.0, "es": 15.5, "it": 15.0,
        "pt": 15.0, "nl": 14.0, "pl": 13.0, "cs": 13.0, "ru": 13.0,
        "tr": 12.0, "hu": 13.0, "ar": 12.0, "hi": 17.0, "ja": 10.0,
        "ko": 10.0, "zh": 6.0, "zh-cn": 6.0, "vi": 16.0, "id": 14.0,
        "ms": 14.0,
    }
    return table.get(lang.lower(), table.get(lang.split("-")[0].lower(), 13.0))


def _estimated_spoken_duration(text: str, lang: str) -> float:
    return len(text or "") / max(1.0, _natural_cps(lang))


def _truncate_for_boundary(
    *,
    chat: Any,
    text: str,
    source: str,
    lang: str,
    slot_seconds: float,
    max_words: int,
    visual_context: dict[str, Any],
    semantics: dict[str, Any],
) -> str | None:
    system = (
        "You are a dubbing line editor. Shorten one translated line so it fits a "
        "hard visual speech boundary. Preserve meaning, names, and language. "
        "Prefer short punchy phrases when pacing is fast. Reply ONLY JSON: "
        "{\"line\":\"...\"}."
    )
    user = json.dumps({
        "target_language": lang,
        "slot_seconds": round(slot_seconds, 3),
        "max_words": max_words,
        "line": text,
        "source_meaning": source,
        "visual_context": visual_context,
        "vision_semantics": semantics,
    }, ensure_ascii=False)
    try:
        reply = chat(system, user)
        data = json.loads(reply or "")
        line = str(data.get("line") or "").strip() if isinstance(data, dict) else ""
    except Exception:
        return None
    return line if line and line != text else None


def _transcript_excerpt(segs: list[dict[str, Any]]) -> str:
    lines = []
    for idx, seg in enumerate(segs[:80]):
        lines.append(f"{idx + 1}. [{float(seg.get('start', 0.0)):.2f}-{float(seg.get('end', 0.0)):.2f}] {seg.get('text', '')}")
    return "\n".join(lines)


def _last_segment_end(segs: list[dict[str, Any]]) -> float:
    return max((float(s.get("end", 0.0)) for s in segs), default=0.0)


def _safe_stem(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:80] or "video"
