#!/usr/bin/env python3
"""Self-learning DeepSeek orchestration layer for the dubbing pipeline.

This module sits above ``dub.py``. It prepares a DeepSeek prompt, injects lessons
from local memory, validates the result, and writes new corrective constraints
when the critic catches a recurring pipeline failure such as false multi-speaker
diarization on a single-speaker source.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


LOGGER = logging.getLogger("self_learning_dub_agent")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"


@dataclass(slots=True)
class FailureMode:
    error_type: str
    trigger: str
    observed_output: dict[str, Any]
    task_metadata: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    occurrences: int = 1


@dataclass(slots=True)
class CorrectiveConstraint:
    error_type: str
    trigger: str
    instruction: str
    payload_patch: dict[str, Any]
    confidence: float = 0.9
    created_at: float = field(default_factory=time.time)
    occurrences: int = 1
    hits: int = 0


@dataclass(slots=True)
class PromptModifier:
    error_type: str
    phrase: str
    created_at: float = field(default_factory=time.time)
    enabled: bool = True


@dataclass(slots=True)
class SystemMemory:
    version: int = 1
    failure_modes: list[FailureMode] = field(default_factory=list)
    corrective_constraints: list[CorrectiveConstraint] = field(default_factory=list)
    system_prompt_modifiers: list[PromptModifier] = field(default_factory=list)
    successful_configurations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemMemory":
        return cls(
            version=int(data.get("version", 1)),
            failure_modes=[
                FailureMode(**item) for item in data.get("failure_modes", [])
                if isinstance(item, dict)
            ],
            corrective_constraints=[
                CorrectiveConstraint(**item)
                for item in data.get("corrective_constraints", [])
                if isinstance(item, dict)
            ],
            system_prompt_modifiers=[
                PromptModifier(**item)
                for item in data.get("system_prompt_modifiers", [])
                if isinstance(item, dict)
            ],
            successful_configurations=list(data.get("successful_configurations", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DubTask:
    task_id: str
    source_text: str
    target_language: str
    metadata: dict[str, Any]
    expected_speakers: int = 1


@dataclass(slots=True)
class AgentConfig:
    enforce_single_speaker: bool = False
    primary_speaker_id: str = "Speaker_1"
    diarization_sensitivity: float = 0.5
    prompt_modifiers: list[str] = field(default_factory=list)

    def speaker_override(self) -> dict[str, Any]:
        return {
            "enforce_single_speaker": self.enforce_single_speaker,
            "primary_speaker_id": self.primary_speaker_id,
            "diarization_sensitivity": self.diarization_sensitivity,
        }


@dataclass(slots=True)
class DubOutput:
    translated_text: str
    summary: str
    speaker_ids: list[str]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationIssue:
    error_type: str
    message: str
    expected: Any
    actual: Any


@dataclass(slots=True)
class ValidationResult:
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)


@dataclass(slots=True)
class AgentRunResult:
    task_id: str
    output: DubOutput
    validation: ValidationResult
    config: AgentConfig
    learned: list[CorrectiveConstraint]

    def pipeline_payload(self) -> dict[str, Any]:
        return {
            "texts_override": {self.output.raw.get("target_lang", ""): [self.output.translated_text]},
            "speaker_override": self.config.speaker_override(),
        }


class DeepSeekClient(Protocol):
    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Return a JSON object from a DeepSeek-compatible chat model."""


class HTTPDeepSeekClient:
    """Minimal async wrapper for the OpenAI-compatible DeepSeek endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEEPSEEK_MODEL,
        url: str = DEEPSEEK_URL,
        timeout_seconds: int = 120,
    ) -> None:
        self.api_key = (api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        self.model = model
        self.url = url
        self.timeout_seconds = timeout_seconds

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for HTTPDeepSeekClient")
        return await asyncio.to_thread(
            self._chat_json_blocking, system_prompt, user_prompt, temperature
        )

    def _chat_json_blocking(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "stream": False,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            data = json.loads(content)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"DeepSeek request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("DeepSeek returned JSON that is not an object")
        return data


class MockDeepSeekClient:
    """Deterministic mock that demonstrates the learning loop without an API key."""

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        await asyncio.sleep(0)
        if "root cause analyst" in system_prompt:
            return {
                "root_cause": (
                    "The upstream task metadata said there was one speaker, but the "
                    "diarization stage still allowed clustering to split vocal style "
                    "changes into multiple speaker IDs."
                ),
                "new_constraint": (
                    "If input metadata says 1 speaker, hard-code "
                    "enforce_single_speaker=True and override downstream clustering "
                    "thresholds."
                ),
                "prompt_modifier": (
                    "The source metadata declares a single speaker; preserve one "
                    "speaker identity and do not invent additional speaker IDs."
                ),
                "payload_patch": {
                    "speaker_override": {
                        "enforce_single_speaker": True,
                        "primary_speaker_id": "Speaker_1",
                        "diarization_sensitivity": 0.0,
                    }
                },
            }

        force_single = "enforce_single_speaker=True" in system_prompt
        if not force_single and "single speaker" not in system_prompt.lower():
            speaker_ids = ["Speaker_1", "Speaker_2"]
        else:
            speaker_ids = ["Speaker_1"]
        return {
            "translated_text": "This is a natural dubbed translation.",
            "summary": "A concise abstraction of the source video.",
            "speaker_ids": speaker_ids,
        }


class JsonMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def load(self) -> SystemMemory:
        return await asyncio.to_thread(self._load_sync)

    async def save(self, memory: SystemMemory) -> None:
        await asyncio.to_thread(self._save_sync, memory)

    def _load_sync(self) -> SystemMemory:
        if not self.path.exists():
            LOGGER.info("Memory file missing; creating a fresh memory matrix at %s", self.path)
            return SystemMemory()
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return SystemMemory.from_dict(data)

    def _save_sync(self, memory: SystemMemory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(memory.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        tmp_path.replace(self.path)


class SelfLearningDubAgent:
    def __init__(self, client: DeepSeekClient, memory_store: JsonMemoryStore) -> None:
        self.client = client
        self.memory_store = memory_store

    async def run(self, task: DubTask) -> AgentRunResult:
        LOGGER.info("Phase A: analyzing task %s", task.task_id)
        memory = await self.memory_store.load()
        config = self._optimize_payload(task, memory)
        system_prompt = self._build_system_prompt(task, config)

        LOGGER.info(
            "Phase B: executing DeepSeek call with speaker_override=%s",
            config.speaker_override(),
        )
        raw_output = await self.client.chat_json(
            system_prompt,
            self._build_user_prompt(task),
            temperature=0.2,
        )
        output = self._parse_output(raw_output, task.target_language)

        LOGGER.info("Phase C: validating output for task %s", task.task_id)
        validation = self.validate_output(task, output)
        learned: list[CorrectiveConstraint] = []
        if validation.passed:
            LOGGER.info("Critic passed: output satisfies speaker and content constraints")
            memory.successful_configurations.append(
                {
                    "task_id": task.task_id,
                    "metadata": task.metadata,
                    "speaker_override": config.speaker_override(),
                    "created_at": time.time(),
                }
            )
            await self.memory_store.save(memory)
        else:
            LOGGER.warning("Critic failed with %d issue(s)", len(validation.issues))
            learned = await self._learn_from_failure(task, output, validation, memory)
            await self.memory_store.save(memory)
            LOGGER.info("Phase D: memory updated with %d new constraint(s)", len(learned))

        return AgentRunResult(
            task_id=task.task_id,
            output=output,
            validation=validation,
            config=config,
            learned=learned,
        )

    def _optimize_payload(self, task: DubTask, memory: SystemMemory) -> AgentConfig:
        config = AgentConfig()
        related_constraints = [
            item for item in memory.corrective_constraints
            if self._constraint_applies(task, item)
        ]
        related_modifiers = [
            item.phrase for item in memory.system_prompt_modifiers
            if item.enabled and self._modifier_applies(task, item)
        ]

        if related_constraints:
            LOGGER.info(
                "Optimization Layer: found %d historical corrective constraint(s)",
                len(related_constraints),
            )
        for constraint in related_constraints:
            patch = constraint.payload_patch.get("speaker_override", {})
            if patch.get("enforce_single_speaker"):
                config.enforce_single_speaker = True
                config.primary_speaker_id = str(
                    patch.get("primary_speaker_id") or config.primary_speaker_id
                )
                config.diarization_sensitivity = float(
                    patch.get("diarization_sensitivity", 0.0)
                )
                constraint.hits += 1

        if task.expected_speakers == 1 and config.enforce_single_speaker:
            related_modifiers.append(
                "Runtime payload already enforces one speaker: "
                "enforce_single_speaker=True."
            )
        config.prompt_modifiers = related_modifiers
        return config

    def _constraint_applies(self, task: DubTask, constraint: CorrectiveConstraint) -> bool:
        if constraint.error_type == "multi_voice_anomaly":
            return task.expected_speakers == 1 or task.metadata.get("speaker_count") == 1
        return constraint.trigger.lower() in json.dumps(task.metadata).lower()

    def _modifier_applies(self, task: DubTask, modifier: PromptModifier) -> bool:
        if modifier.error_type == "multi_voice_anomaly":
            return task.expected_speakers == 1 or task.metadata.get("speaker_count") == 1
        return True

    def _build_system_prompt(self, task: DubTask, config: AgentConfig) -> str:
        lessons = "\n".join(f"- {phrase}" for phrase in config.prompt_modifiers)
        if not lessons:
            lessons = "- No prior lessons matched this task."
        return (
            "You are the Execution Layer for a video translation and abstraction "
            "agent. Translate naturally, summarize faithfully, and preserve speaker "
            "identity constraints.\n\n"
            f"Target language: {task.target_language}\n"
            f"Expected speakers: {task.expected_speakers}\n"
            f"Optimization payload: enforce_single_speaker={config.enforce_single_speaker}, "
            f"primary_speaker_id={config.primary_speaker_id}, "
            f"diarization_sensitivity={config.diarization_sensitivity}\n\n"
            "Lessons learned from memory:\n"
            f"{lessons}\n\n"
            "Reply with ONLY JSON containing translated_text, summary, and speaker_ids."
        )

    def _build_user_prompt(self, task: DubTask) -> str:
        return json.dumps(
            {
                "task_id": task.task_id,
                "source_text": task.source_text,
                "metadata": task.metadata,
            },
            ensure_ascii=True,
        )

    def _parse_output(self, raw: dict[str, Any], target_language: str) -> DubOutput:
        speaker_ids = raw.get("speaker_ids") or []
        if not isinstance(speaker_ids, list):
            speaker_ids = [str(speaker_ids)]
        raw["target_lang"] = target_language
        return DubOutput(
            translated_text=str(raw.get("translated_text", "")).strip(),
            summary=str(raw.get("summary", "")).strip(),
            speaker_ids=[str(item).strip() for item in speaker_ids if str(item).strip()],
            raw=raw,
        )

    def validate_output(self, task: DubTask, output: DubOutput) -> ValidationResult:
        issues: list[ValidationIssue] = []
        actual_speakers = len(set(output.speaker_ids))
        if actual_speakers > task.expected_speakers:
            issues.append(
                ValidationIssue(
                    error_type="multi_voice_anomaly",
                    message=(
                        "Generated speaker IDs exceed source metadata; this would "
                        "create extra voices downstream."
                    ),
                    expected=task.expected_speakers,
                    actual=actual_speakers,
                )
            )
        if not output.translated_text:
            issues.append(
                ValidationIssue(
                    error_type="empty_translation",
                    message="Translation text is empty.",
                    expected="non-empty translated_text",
                    actual=output.translated_text,
                )
            )
        if not output.summary:
            issues.append(
                ValidationIssue(
                    error_type="empty_summary",
                    message="Summary text is empty.",
                    expected="non-empty summary",
                    actual=output.summary,
                )
            )
        return ValidationResult(passed=not issues, issues=issues)

    async def _learn_from_failure(
        self,
        task: DubTask,
        output: DubOutput,
        validation: ValidationResult,
        memory: SystemMemory,
    ) -> list[CorrectiveConstraint]:
        learned: list[CorrectiveConstraint] = []
        for issue in validation.issues:
            LOGGER.warning(
                "Learning Module: asking DeepSeek to diagnose %s", issue.error_type
            )
            reflection = await self.client.chat_json(
                "You are a root cause analyst for a self-correcting video dubbing "
                "pipeline. Return JSON with root_cause, new_constraint, "
                "prompt_modifier, and payload_patch.",
                json.dumps(
                    {
                        "issue": asdict(issue),
                        "task_metadata": task.metadata,
                        "output": asdict(output),
                    },
                    ensure_ascii=True,
                ),
                temperature=0.1,
            )
            failure = FailureMode(
                error_type=issue.error_type,
                trigger=self._trigger_for_issue(task, issue),
                observed_output=asdict(output),
                task_metadata=task.metadata,
            )
            memory.failure_modes.append(failure)

            constraint = self._constraint_from_reflection(issue, reflection)
            if not self._has_equivalent_constraint(memory, constraint):
                memory.corrective_constraints.append(constraint)
                learned.append(constraint)
                LOGGER.warning(
                    "Memory Layer: wrote corrective constraint: %s",
                    constraint.instruction,
                )
            prompt_phrase = str(reflection.get("prompt_modifier") or "").strip()
            if prompt_phrase and not self._has_prompt_modifier(memory, issue.error_type, prompt_phrase):
                memory.system_prompt_modifiers.append(
                    PromptModifier(error_type=issue.error_type, phrase=prompt_phrase)
                )
                LOGGER.info("Memory Layer: added prompt modifier for %s", issue.error_type)
        return learned

    def _trigger_for_issue(self, task: DubTask, issue: ValidationIssue) -> str:
        if issue.error_type == "multi_voice_anomaly":
            return "metadata.expected_speakers == 1 and generated_speaker_ids > 1"
        return f"{issue.error_type}:{task.task_id}"

    def _constraint_from_reflection(
        self, issue: ValidationIssue, reflection: dict[str, Any]
    ) -> CorrectiveConstraint:
        if issue.error_type == "multi_voice_anomaly":
            payload_patch = {
                "speaker_override": {
                    "enforce_single_speaker": True,
                    "primary_speaker_id": "Speaker_1",
                    "diarization_sensitivity": 0.0,
                }
            }
            payload_patch.update(reflection.get("payload_patch") or {})
            return CorrectiveConstraint(
                error_type=issue.error_type,
                trigger="metadata.expected_speakers == 1 and generated_speaker_ids > 1",
                instruction=(
                    str(reflection.get("new_constraint") or "").strip()
                    or "If input metadata says 1 speaker, hard-code "
                    "enforce_single_speaker=True and override downstream clustering thresholds."
                ),
                payload_patch=payload_patch,
                confidence=0.98,
            )
        return CorrectiveConstraint(
            error_type=issue.error_type,
            trigger=issue.message,
            instruction=str(reflection.get("new_constraint") or issue.message),
            payload_patch=reflection.get("payload_patch") or {},
            confidence=0.75,
        )

    def _has_equivalent_constraint(
        self, memory: SystemMemory, new_constraint: CorrectiveConstraint
    ) -> bool:
        for item in memory.corrective_constraints:
            if (
                item.error_type == new_constraint.error_type
                and item.trigger == new_constraint.trigger
                and item.payload_patch == new_constraint.payload_patch
            ):
                item.occurrences += 1
                return True
        return False

    def _has_prompt_modifier(
        self, memory: SystemMemory, error_type: str, phrase: str
    ) -> bool:
        return any(
            item.error_type == error_type and item.phrase == phrase
            for item in memory.system_prompt_modifiers
        )


def _metadata_from_job_config(cfg: dict[str, Any]) -> dict[str, Any]:
    speaker_override = cfg.get("speaker_override")
    if not isinstance(speaker_override, dict):
        speaker_override = {}
    metadata: dict[str, Any] = {
        "preset": cfg.get("preset") or "",
        "speakers_requested": bool(cfg.get("speakers")),
        "gender_requested": bool(cfg.get("gender")),
        "clone_requested": bool(cfg.get("clone")),
        "speaker_override": speaker_override,
    }
    if speaker_override.get("enforce_single_speaker"):
        metadata["speaker_count"] = 1
    if "expected_speakers" in cfg:
        metadata["speaker_count"] = cfg.get("expected_speakers")
    return metadata


def optimize_dub_job_config(
    cfg: dict[str, Any],
    memory_path: str | Path | None = None,
    task_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply learned memory constraints to a dub.py jobs config.

    The GUI and browser app both call this before writing ``_jobs.json``. It keeps
    the learning layer out of the heavy video runtime while still injecting the
    concrete payload field that ``dub.run_one()`` already understands:
    ``speaker_override``.
    """
    memory_file = Path(memory_path) if memory_path else Path(__file__).with_name("system_memory.json")
    memory = JsonMemoryStore(memory_file)._load_sync()
    metadata = _metadata_from_job_config(cfg)
    metadata.update(task_metadata or {})
    expected = int(metadata.get("speaker_count") or metadata.get("expected_speakers") or 0)
    task = DubTask(
        task_id=str(cfg.get("job_id") or "dub-app-job"),
        source_text="",
        target_language=str(cfg.get("langs") or ""),
        expected_speakers=expected if expected > 0 else 99,
        metadata=metadata,
    )
    agent = SelfLearningDubAgent(MockDeepSeekClient(), JsonMemoryStore(memory_file))
    config = agent._optimize_payload(task, memory)
    if config.enforce_single_speaker:
        patched = dict(cfg)
        patched["speaker_override"] = config.speaker_override()
        patched["speakers"] = False
        patched["gender"] = False
        patched["clone"] = bool(cfg.get("clone")) and not bool(patched.get("speakers"))
        patched["_learning_applied"] = {
            "error_type": "multi_voice_anomaly",
            "speaker_override": config.speaker_override(),
        }
        LOGGER.info(
            "Optimization Layer: applied learned speaker override to dub job: %s",
            config.speaker_override(),
        )
        return patched
    return cfg


async def demo(memory_path: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    agent = SelfLearningDubAgent(MockDeepSeekClient(), JsonMemoryStore(memory_path))
    task = DubTask(
        task_id="demo-single-speaker-recap",
        source_text="I walked into the room and explained what happened next.",
        target_language="en",
        expected_speakers=1,
        metadata={"speaker_count": 1, "confidence": 0.82, "content_type": "recap"},
    )

    LOGGER.info("Demo run 1: memory is allowed to learn from the failure")
    first = await agent.run(task)
    LOGGER.info("Demo run 1 validation passed=%s", first.validation.passed)

    LOGGER.info("Demo run 2: optimization layer should inject the learned override")
    second = await agent.run(task)
    LOGGER.info("Demo run 2 validation passed=%s", second.validation.passed)
    LOGGER.info("Final optimized payload: %s", second.pipeline_payload())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the self-learning dub agent demo.")
    parser.add_argument(
        "--fresh-demo",
        action="store_true",
        help="Use a throwaway empty memory file to show failure, learning, then correction.",
    )
    args = parser.parse_args()
    path = (
        Path(tempfile.gettempdir()) / f"self_learning_dub_demo_{os.getpid()}.json"
        if args.fresh_demo
        else Path(__file__).with_name("system_memory.json")
    )
    asyncio.run(demo(path))
