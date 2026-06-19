#!/usr/bin/env python3
"""Human feedback storage for the dubbing editor and learning loop."""

from __future__ import annotations

import json
import time
import uuid
import re
from pathlib import Path
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
except Exception:  # pragma: no cover - only used when optional dependency is absent.
    BaseModel = object  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    ValidationError = ValueError  # type: ignore[assignment]
    field_validator = None  # type: ignore[assignment]


ErrorCategory = Literal["timing_mismatch", "wrong_tone", "mistranslation"]


if Field is not None:
    class ContextFingerprint(BaseModel):
        detected_pace: str = "unknown"
        visual_scene_type: str = "unknown"
        original_tone: str = "unknown"


    class HumanCorrection(BaseModel):
        action: str = Field(min_length=1)
        injected_instruction: str = Field(min_length=1)


    class GeneralizedPreferenceAnnotation(BaseModel):
        error_id: str = Field(default_factory=lambda: f"err_{uuid.uuid4().hex[:8]}")
        context_fingerprint: ContextFingerprint = Field(default_factory=ContextFingerprint)
        human_correction: HumanCorrection
        source_annotation_id: str = ""
        created_at: float = Field(default_factory=time.time)


    class CorrectionAnnotation(BaseModel):
        """Pydantic contract for one user correction on a dubbed video span."""

        video_id: str = Field(min_length=1)
        timestamp_start: float = Field(ge=0)
        timestamp_end: float = Field(ge=0)
        error_category: ErrorCategory
        original_output: str = ""
        corrected_output: str = ""
        user_annotation_notes: str = ""
        resolved_state: bool = False
        project_id: str = ""
        segment_index: int | None = None
        source_text: str = ""
        target_language: str = ""
        visual_context: dict[str, Any] = Field(default_factory=dict)
        context_fingerprint: dict[str, Any] = Field(default_factory=dict)
        human_correction: dict[str, Any] = Field(default_factory=dict)
        created_at: float = Field(default_factory=time.time)
        annotation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)

        @field_validator("timestamp_end")
        @classmethod
        def _end_after_start(cls, value: float, info: Any) -> float:
            start = info.data.get("timestamp_start", 0.0)
            if value < start:
                raise ValueError("timestamp_end must be greater than or equal to timestamp_start")
            return value
else:
    class GeneralizedPreferenceAnnotation:  # type: ignore[no-redef]
        def __init__(self, **data: Any) -> None:
            self.error_id = str(data.get("error_id") or f"err_{uuid.uuid4().hex[:8]}")
            fp = data.get("context_fingerprint") if isinstance(data.get("context_fingerprint"), dict) else {}
            hc = data.get("human_correction") if isinstance(data.get("human_correction"), dict) else {}
            self.context_fingerprint = {
                "detected_pace": str(fp.get("detected_pace") or "unknown"),
                "visual_scene_type": str(fp.get("visual_scene_type") or "unknown"),
                "original_tone": str(fp.get("original_tone") or "unknown"),
            }
            self.human_correction = {
                "action": str(hc.get("action") or "").strip(),
                "injected_instruction": str(hc.get("injected_instruction") or "").strip(),
            }
            if not self.human_correction["action"] or not self.human_correction["injected_instruction"]:
                raise ValueError("human_correction.action and injected_instruction are required")
            self.source_annotation_id = str(data.get("source_annotation_id") or "")
            self.created_at = float(data.get("created_at") or time.time())

        def model_dump(self, mode: str = "python") -> dict[str, Any]:
            return dict(self.__dict__)


    class CorrectionAnnotation:  # type: ignore[no-redef]
        """Small fallback that preserves the same data shape when pydantic is absent."""

        _allowed = {"timing_mismatch", "wrong_tone", "mistranslation"}

        def __init__(self, **data: Any) -> None:
            self.video_id = str(data.get("video_id") or "").strip()
            self.timestamp_start = float(data.get("timestamp_start", 0.0))
            self.timestamp_end = float(data.get("timestamp_end", self.timestamp_start))
            self.error_category = str(data.get("error_category") or "")
            self.original_output = str(data.get("original_output") or "")
            self.corrected_output = str(data.get("corrected_output") or "")
            self.user_annotation_notes = str(data.get("user_annotation_notes") or "")
            self.resolved_state = bool(data.get("resolved_state", False))
            self.project_id = str(data.get("project_id") or "")
            self.segment_index = data.get("segment_index")
            self.source_text = str(data.get("source_text") or "")
            self.target_language = str(data.get("target_language") or "")
            self.visual_context = data.get("visual_context") if isinstance(data.get("visual_context"), dict) else {}
            self.context_fingerprint = data.get("context_fingerprint") if isinstance(data.get("context_fingerprint"), dict) else {}
            self.human_correction = data.get("human_correction") if isinstance(data.get("human_correction"), dict) else {}
            self.created_at = float(data.get("created_at") or time.time())
            self.annotation_id = str(data.get("annotation_id") or uuid.uuid4().hex)
            if not self.video_id:
                raise ValueError("video_id is required")
            if self.timestamp_start < 0 or self.timestamp_end < self.timestamp_start:
                raise ValueError("invalid timestamp range")
            if self.error_category not in self._allowed:
                raise ValueError("invalid error_category")

        def model_dump(self, mode: str = "python") -> dict[str, Any]:
            return dict(self.__dict__)


class FeedbackStorage:
    """Append-only store that also emits preference examples for later tuning."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.annotations_path = self.root / "corrections.jsonl"
        self.preference_path = self.root / "preferences.jsonl"
        self.generalized_preference_path = self.root / "generalized_preferences.jsonl"

    def add(self, payload: dict[str, Any]) -> CorrectionAnnotation:
        item = CorrectionAnnotation(**payload)
        row = item.model_dump(mode="json") if hasattr(item, "model_dump") else item.__dict__
        self._append_jsonl(self.annotations_path, row)
        pref = self._preference_pair(row)
        if pref:
            self._append_jsonl(self.preference_path, pref)
        generalized = self._generalized_preference(row)
        if generalized:
            self._append_jsonl(self.generalized_preference_path, generalized)
        return item

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.annotations_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.annotations_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows[-max(1, int(limit)):]

    def lessons(self, category: str | None = None, limit: int = 12) -> list[str]:
        lessons: list[str] = []
        for row in reversed(self.recent(limit=200)):
            if category and row.get("error_category") != category:
                continue
            note = str(row.get("user_annotation_notes") or "").strip()
            if note and note not in lessons:
                lessons.append(note)
            generalized = self._matching_generalized(row)
            if generalized:
                instruction = str(
                    (generalized.get("human_correction") or {}).get("injected_instruction") or ""
                ).strip()
                if instruction and instruction not in lessons:
                    lessons.append(instruction)
            if len(lessons) >= limit:
                break
        return lessons

    def generalized_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.generalized_preference_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.generalized_preference_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows[-max(1, int(limit)):]

    def _preference_pair(self, row: dict[str, Any]) -> dict[str, Any] | None:
        rejected = str(row.get("original_output") or "").strip()
        chosen = str(row.get("corrected_output") or "").strip()
        notes = str(row.get("user_annotation_notes") or "").strip()
        if not chosen and not notes:
            return None
        prompt = {
            "source_text": row.get("source_text", ""),
            "target_language": row.get("target_language", ""),
            "timestamp_start": row.get("timestamp_start"),
            "timestamp_end": row.get("timestamp_end"),
            "error_category": row.get("error_category"),
            "visual_context": row.get("visual_context") or {},
            "instruction": notes,
        }
        return {
            "annotation_id": row.get("annotation_id"),
            "video_id": row.get("video_id"),
            "prompt": prompt,
            "chosen": chosen or rejected,
            "rejected": rejected,
            "created_at": row.get("created_at", time.time()),
        }

    def _generalized_preference(self, row: dict[str, Any]) -> dict[str, Any] | None:
        direct_fp = row.get("context_fingerprint") if isinstance(row.get("context_fingerprint"), dict) else {}
        direct_hc = row.get("human_correction") if isinstance(row.get("human_correction"), dict) else {}
        fingerprint = direct_fp or self._fingerprint_from_row(row)
        correction = direct_hc or self._correction_from_row(row)
        if not correction.get("action") or not correction.get("injected_instruction"):
            return None
        try:
            item = GeneralizedPreferenceAnnotation(
                error_id=str(row.get("error_id") or f"err_{uuid.uuid4().hex[:8]}"),
                context_fingerprint=fingerprint,
                human_correction=correction,
                source_annotation_id=str(row.get("annotation_id") or ""),
            )
            return item.model_dump(mode="json") if hasattr(item, "model_dump") else item.__dict__
        except Exception:
            return None

    def _fingerprint_from_row(self, row: dict[str, Any]) -> dict[str, Any]:
        vc = row.get("visual_context") if isinstance(row.get("visual_context"), dict) else {}
        semantics = vc.get("vision_semantics") if isinstance(vc.get("vision_semantics"), dict) else {}
        pacing = str(semantics.get("pacing") or vc.get("pacing") or "").lower()
        expression = str(vc.get("speaker_expression") or vc.get("visual_context") or "").lower()
        start = float(row.get("timestamp_start") or 0.0)
        end = float(row.get("timestamp_end") or start)
        duration = max(0.0, end - start)
        text = str(row.get("original_output") or row.get("corrected_output") or "")
        words_per_second = len(re.findall(r"\S+", text)) / duration if duration > 0 else 0.0
        if not pacing:
            pacing = "fast" if words_per_second > 3.5 else "slow" if words_per_second < 1.4 else "moderate"
        scene_type = str(
            semantics.get("visual_scene_type")
            or vc.get("visual_scene_type")
            or _infer_scene_type(vc)
        )
        tone = "high_energy" if any(x in expression for x in ("excited", "shout", "energetic", "angry", "intense")) else (
            "quiet" if any(x in expression for x in ("calm", "quiet", "soft", "serious")) else "unknown"
        )
        return {
            "detected_pace": pacing or "unknown",
            "visual_scene_type": scene_type or "unknown",
            "original_tone": tone,
        }

    def _correction_from_row(self, row: dict[str, Any]) -> dict[str, Any]:
        category = str(row.get("error_category") or "")
        notes = str(row.get("user_annotation_notes") or "").strip()
        original = str(row.get("original_output") or "")
        corrected = str(row.get("corrected_output") or "")
        if category == "timing_mismatch":
            action = "reduce_speed_and_truncate"
        elif category == "wrong_tone":
            action = "adjust_tone"
        else:
            action = "preserve_meaning"
        if notes:
            instruction = notes
        elif corrected and original and len(corrected) < len(original):
            instruction = "When this context recurs, prefer shorter phrasing that preserves the core meaning."
        else:
            instruction = "Apply the human corrected wording pattern when similar pacing, tone, and scene context recur."
        return {"action": action, "injected_instruction": instruction}

    def _matching_generalized(self, row: dict[str, Any]) -> dict[str, Any] | None:
        fingerprint = self._fingerprint_from_row(row)
        for pref in reversed(self.generalized_recent(limit=200)):
            fp = pref.get("context_fingerprint") or {}
            if (
                fp.get("detected_pace") == fingerprint.get("detected_pace")
                and fp.get("visual_scene_type") == fingerprint.get("visual_scene_type")
                and fp.get("original_tone") == fingerprint.get("original_tone")
            ):
                return pref
        return None

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _infer_scene_type(visual_context: dict[str, Any]) -> str:
    blob = json.dumps(visual_context, ensure_ascii=False).lower()
    if any(term in blob for term in ("close-up", "close up", "face", "talking head")):
        return "close_up_talking"
    if any(term in blob for term in ("wide", "landscape", "room", "street")):
        return "wide_context"
    if any(term in blob for term in ("screen", "slide", "caption", "document")):
        return "screen_or_text"
    return "unknown"
