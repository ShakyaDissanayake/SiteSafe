"""Safety Compliance Rule Engine for Construction-PPE dataset.

Evaluates per-worker PPE states for datasets that provide:
  - Person
  - helmet / no_helmet
  - gloves / no_gloves
  - vest
  - boots / no_boots
  - goggles / no_goggle
  - none (no PPE indicator)
"""

import json
from pathlib import Path
from typing import Optional

from inference import (
    ComplianceReport,
    SafetyRule,
    SceneVerdict,
    Severity,
    Violation,
    WorkerPPEState,
    ZoneType,
)


LOW_LIGHT_BRIGHTNESS = 60.0
LOW_CONFIDENCE_THRESHOLD = 0.45


class ComplianceEngine:
    """Rule-based PPE compliance evaluation engine."""

    def __init__(
        self,
        ruleset: list[SafetyRule],
        zone_map: Optional[dict] = None,
    ) -> None:
        self.rules = ruleset
        self.zone_map = zone_map

    @classmethod
    def from_json(
        cls,
        json_path: str,
        zone_map: Optional[dict] = None,
    ) -> "ComplianceEngine":
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rules = []
        for r in data["rules"]:
            rules.append(
                SafetyRule(
                    rule_id=r["rule_id"],
                    rule_name=r["rule_name"],
                    description=r["description"],
                    detection_targets=r["detection_targets"],
                    severity=Severity(r["severity"]),
                    zone_applicability=ZoneType(r.get("zone_applicability", "ALL")),
                    suggested_action=r["suggested_action"],
                    osha_reference=r.get("osha_reference", ""),
                )
            )
        return cls(rules, zone_map)

    def evaluate(
        self,
        worker_states: list[WorkerPPEState],
        frame_metadata: dict,
    ) -> ComplianceReport:
        all_violations: list[Violation] = []
        low_confidence_flags: list[str] = []
        compliant_count = 0

        for state in worker_states:
            worker_violations = self._evaluate_worker(state, frame_metadata)
            uncertainty = self._check_uncertainty(state)
            if uncertainty:
                low_confidence_flags.append(uncertainty)

            if worker_violations:
                all_violations.extend(worker_violations)
            else:
                compliant_count += 1

        verdict = self._compute_scene_verdict(all_violations)
        confidence = self._compute_overall_confidence(worker_states, all_violations)

        return ComplianceReport(
            frame_id=frame_metadata.get("frame_id", "unknown"),
            timestamp=frame_metadata.get("timestamp", ""),
            scene_verdict=verdict.value,
            overall_confidence=confidence,
            worker_count=len(worker_states),
            compliant_workers=compliant_count,
            violation_count=len(all_violations),
            violations=all_violations,
            scene_brightness=frame_metadata.get("brightness", 128.0),
            low_confidence_flags=low_confidence_flags,
        )

    def _evaluate_worker(
        self,
        state: WorkerPPEState,
        frame_metadata: dict,
    ) -> list[Violation]:
        violations = []

        for rule in self.rules:
            violated, confidence, desc = self._check_rule(rule, state, frame_metadata)
            if violated:
                violations.append(
                    Violation(
                        worker_id=state.worker_id,
                        rule_id=rule.rule_id,
                        rule_name=rule.rule_name,
                        severity=rule.severity.value,
                        description=desc,
                        confidence=confidence,
                        bbox=state.worker_bbox.as_tuple(),
                        suggested_action=rule.suggested_action,
                    )
                )

        return violations

    def _check_rule(
        self,
        rule: SafetyRule,
        state: WorkerPPEState,
        metadata: dict,
    ) -> tuple[bool, float, str]:
        rule_checks = {
            "PPE-001": self._check_hardhat_required,
            "PPE-002": self._check_gloves_required,
            "PPE-003": self._check_vest_required,
            "PPE-004": self._check_boots_required,
            "PPE-005": self._check_goggles_required,
            "PPE-006": self._check_full_ppe,
            "PPE-007": self._check_none_class,
            "ENV-001": self._check_low_visibility,
        }

        check_fn = rule_checks.get(rule.rule_id)
        if check_fn is None:
            return False, 0.0, ""
        return check_fn(state, metadata)

    @staticmethod
    def _check_hardhat_required(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        if state.is_distant:
            return False, 0.0, ""
        if not state.has_helmet:
            conf = max(state.worker_confidence, state.helmet_confidence)
            return True, conf, f"Worker #{state.worker_id} is not wearing a helmet."
        return False, 0.0, ""

    @staticmethod
    def _check_gloves_required(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        if state.is_distant:
            return False, 0.0, ""
        if not state.has_gloves:
            conf = max(state.worker_confidence, state.gloves_confidence)
            return True, conf, f"Worker #{state.worker_id} is not wearing gloves."
        return False, 0.0, ""

    @staticmethod
    def _check_vest_required(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        if state.is_distant:
            return False, 0.0, ""
        if not state.has_vest:
            conf = max(state.worker_confidence, state.vest_confidence)
            return True, conf, f"Worker #{state.worker_id} is not wearing a high-visibility vest."
        return False, 0.0, ""

    @staticmethod
    def _check_boots_required(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        if state.is_distant:
            return False, 0.0, ""
        if not state.has_boots:
            conf = max(state.worker_confidence, state.boots_confidence)
            return True, conf, f"Worker #{state.worker_id} is not wearing safety boots."
        return False, 0.0, ""

    @staticmethod
    def _check_goggles_required(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        if state.is_distant:
            return False, 0.0, ""
        if not state.has_goggles:
            conf = max(state.worker_confidence, state.goggles_confidence)
            return True, conf, f"Worker #{state.worker_id} is not wearing eye protection goggles."
        return False, 0.0, ""

    @staticmethod
    def _check_full_ppe(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        if state.is_distant:
            return False, 0.0, ""

        missing = []
        if not state.has_helmet:
            missing.append("helmet")
        if not state.has_gloves:
            missing.append("gloves")
        if not state.has_vest:
            missing.append("vest")
        if not state.has_boots:
            missing.append("boots")
        if not state.has_goggles:
            missing.append("goggles")

        if missing:
            conf = state.worker_confidence
            items = ", ".join(missing)
            return True, conf, f"Worker #{state.worker_id} is missing required PPE items: {items}."
        return False, 0.0, ""

    @staticmethod
    def _check_none_class(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        if state.none_class_detected:
            return (
                True,
                state.worker_confidence,
                f"Worker #{state.worker_id} matched the 'none' class indicating no detectable PPE.",
            )
        return False, 0.0, ""

    @staticmethod
    def _check_low_visibility(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        brightness = metadata.get("brightness", 128.0)
        if brightness >= LOW_LIGHT_BRIGHTNESS:
            return False, 0.0, ""
        if not state.has_vest:
            conf = max(state.worker_confidence, state.vest_confidence)
            return (
                True,
                conf,
                f"Worker #{state.worker_id} has no vest in low visibility (brightness={brightness:.0f}/255).",
            )
        return False, 0.0, ""

    @staticmethod
    def _compute_scene_verdict(violations: list[Violation]) -> SceneVerdict:
        if not violations:
            return SceneVerdict.SAFE

        has_critical = any(v.severity == Severity.CRITICAL.value for v in violations)
        if has_critical:
            return SceneVerdict.UNSAFE
        return SceneVerdict.WARNING

    @staticmethod
    def _compute_overall_confidence(
        states: list[WorkerPPEState],
        violations: list[Violation],
    ) -> float:
        confidences = [s.worker_confidence for s in states]
        confidences.extend(v.confidence for v in violations)
        if not confidences:
            return 0.0
        return float(sum(confidences) / len(confidences))

    @staticmethod
    def _check_uncertainty(state: WorkerPPEState) -> Optional[str]:
        if state.is_occluded:
            return f"Worker #{state.worker_id}: occluded; PPE status may be uncertain."
        if state.is_distant:
            return (
                f"Worker #{state.worker_id}: too distant ({state.worker_bbox.height}px); "
                "fine-grained PPE attributes may be unreliable."
            )
        if state.worker_confidence < LOW_CONFIDENCE_THRESHOLD:
            return (
                f"Worker #{state.worker_id}: low detection confidence "
                f"({state.worker_confidence:.2f})."
            )
        return None
