"""Safety Compliance Rule Engine.

Evaluates worker PPE states against the safety ruleset and produces
per-worker violations with confidence scores and human-readable explanations.

The engine loads rules from the JSON ruleset and applies them based on:
  - Worker PPE state (helmet, vest, harness presence)
  - Zone context (machinery proximity, height zone, danger zone)
  - Scene conditions (low-light, overall brightness)
  - Edge cases (occluded workers, distant workers)

Typical usage:
    engine = ComplianceEngine.from_json("rules/safety_rules.json")
    report = engine.evaluate(worker_states, frame_metadata)
"""

import json
from pathlib import Path
from typing import Optional

from inference import (
    BBox,
    ComplianceReport,
    SafetyRule,
    SceneVerdict,
    Severity,
    Violation,
    WorkerPPEState,
    ZoneType,
)


# ── Constants ────────────────────────────────────────────────────────────
LOW_LIGHT_BRIGHTNESS = 60.0
HIGH_CONFIDENCE_THRESHOLD = 0.55
LOW_CONFIDENCE_THRESHOLD = 0.45
POSTURE_ASPECT_RATIO_THRESHOLD = 1.5  # Width > 1.5× height = horizontal


class ComplianceEngine:
    """Rule-based safety compliance evaluation engine.

    Attributes:
        rules: List of SafetyRule objects loaded from the JSON schema.
        zone_map: Optional zone configuration for spatial rule filtering.
    """

    def __init__(
        self,
        ruleset: list[SafetyRule],
        zone_map: Optional[dict] = None,
    ) -> None:
        """Initialize the compliance engine.

        Args:
            ruleset: List of SafetyRule objects to evaluate.
            zone_map: Optional zone configuration dict.
        """
        self.rules = ruleset
        self.zone_map = zone_map

    @classmethod
    def from_json(
        cls,
        json_path: str,
        zone_map: Optional[dict] = None,
    ) -> "ComplianceEngine":
        """Create a ComplianceEngine from a JSON ruleset file.

        Args:
            json_path: Path to safety_rules.json.
            zone_map: Optional zone configuration dict.

        Returns:
            Initialized ComplianceEngine instance.
        """
        path = Path(json_path)
        with open(path, "r") as f:
            data = json.load(f)

        rules = []
        for r in data["rules"]:
            rules.append(SafetyRule(
                rule_id=r["rule_id"],
                rule_name=r["rule_name"],
                description=r["description"],
                detection_targets=r["detection_targets"],
                severity=Severity(r["severity"]),
                zone_applicability=ZoneType(r["zone_applicability"]),
                suggested_action=r["suggested_action"],
                osha_reference=r.get("osha_reference", ""),
            ))
        return cls(rules, zone_map)

    def evaluate(
        self,
        worker_states: list[WorkerPPEState],
        frame_metadata: dict,
    ) -> ComplianceReport:
        """Evaluate all workers against the safety ruleset.

        Args:
            worker_states: List of WorkerPPEState from the detector.
            frame_metadata: Dict with keys: frame_id, timestamp,
                brightness, frame_shape.

        Returns:
            ComplianceReport with all violations and scene verdict.
        """
        all_violations: list[Violation] = []
        low_confidence_flags: list[str] = []
        compliant_count = 0

        for state in worker_states:
            worker_violations = self._evaluate_worker(
                state, frame_metadata
            )
            # Handle edge cases — uncertain workers
            uncertainty = self._check_uncertainty(state)
            if uncertainty:
                low_confidence_flags.append(uncertainty)

            if worker_violations:
                all_violations.extend(worker_violations)
            else:
                compliant_count += 1

        # Compute scene verdict and confidence
        verdict = self._compute_scene_verdict(all_violations)
        confidence = self._compute_overall_confidence(
            worker_states, all_violations
        )

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
        """Apply all applicable rules to a single worker.

        Args:
            state: Worker's PPE state.
            frame_metadata: Frame-level metadata.

        Returns:
            List of Violations for this worker.
        """
        violations = []

        for rule in self.rules:
            # Skip rules not applicable to this worker's zone
            if not self._is_zone_applicable(rule, state):
                continue

            # Evaluate each rule by ID
            violated, confidence, desc = self._check_rule(
                rule, state, frame_metadata
            )

            if violated:
                violations.append(Violation(
                    worker_id=state.worker_id,
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    severity=rule.severity.value,
                    description=desc,
                    confidence=confidence,
                    bbox=state.worker_bbox.as_tuple(),
                    suggested_action=rule.suggested_action,
                ))

        return violations

    def _check_rule(
        self,
        rule: SafetyRule,
        state: WorkerPPEState,
        metadata: dict,
    ) -> tuple[bool, float, str]:
        """Evaluate a single rule against a worker state.

        Args:
            rule: Safety rule to check.
            state: Worker's PPE state.
            metadata: Frame metadata.

        Returns:
            Tuple of (is_violated, confidence, description).
        """
        wid = state.worker_id
        rule_checks = {
            "PPE-001": self._check_hardhat_required,
            "PPE-002": self._check_vest_required,
            "PPE-003": self._check_harness_at_height,
            "PPE-004": self._check_improper_helmet,
            "PPE-005": self._check_improper_vest,
            "PPE-006": self._check_complete_ppe,
            "PROX-001": self._check_machinery_proximity,
            "POST-001": self._check_posture_anomaly,
            "SCENE-001": self._check_danger_zone_entry,
            "ENV-001": self._check_low_visibility,
        }

        check_fn = rule_checks.get(rule.rule_id)
        if check_fn is None:
            return False, 0.0, ""

        return check_fn(state, metadata)

    # ── Individual Rule Checks ───────────────────────────────────────

    @staticmethod
    def _check_hardhat_required(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        """PPE-001: Worker must have a hard hat."""
        if state.is_distant:
            return False, 0.0, ""
        if not state.has_helmet:
            conf = max(state.worker_confidence, state.helmet_confidence)
            return True, conf, (
                f"Worker #{state.worker_id} is not wearing a hard hat."
            )
        return False, 0.0, ""

    @staticmethod
    def _check_vest_required(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        """PPE-002: Worker must have a hi-vis vest in active zones."""
        if state.is_distant:
            return False, 0.0, ""
        if not state.has_vest:
            conf = max(state.worker_confidence, state.vest_confidence)
            return True, conf, (
                f"Worker #{state.worker_id} is not wearing a "
                f"high-visibility vest."
            )
        return False, 0.0, ""

    @staticmethod
    def _check_harness_at_height(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        """PPE-003: Worker at height must have a safety harness."""
        if not state.in_height_zone:
            return False, 0.0, ""
        if not state.has_harness:
            conf = state.worker_confidence
            return True, conf, (
                f"Worker #{state.worker_id} is at height without "
                f"a safety harness."
            )
        return False, 0.0, ""

    @staticmethod
    def _check_improper_helmet(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        """PPE-004: Helmet detected but not properly worn."""
        if state.helmet_proper is None or state.helmet_proper:
            return False, 0.0, ""
        conf = state.helmet_confidence
        return True, conf, (
            f"Worker #{state.worker_id} has a helmet but it is "
            f"not properly worn on the head."
        )

    @staticmethod
    def _check_improper_vest(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        """PPE-005: Vest detected but not properly fastened."""
        if state.vest_proper is None or state.vest_proper:
            return False, 0.0, ""
        conf = state.vest_confidence
        return True, conf, (
            f"Worker #{state.worker_id} has a vest but it appears "
            f"unfastened or improperly worn."
        )

    @staticmethod
    def _check_complete_ppe(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        """PPE-006: Worker in active zone must have helmet AND vest."""
        if state.is_distant:
            return False, 0.0, ""
        missing = []
        if not state.has_helmet:
            missing.append("hard hat")
        if not state.has_vest:
            missing.append("hi-vis vest")
        if len(missing) > 0 and len(missing) < 2:
            # Only trigger if partially missing (full missing is PPE-001/002)
            conf = state.worker_confidence
            items = " and ".join(missing)
            return True, conf, (
                f"Worker #{state.worker_id} is missing {items} "
                f"in the active zone."
            )
        return False, 0.0, ""

    @staticmethod
    def _check_machinery_proximity(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        """PROX-001: Worker near machinery must have full PPE."""
        if not state.near_machinery:
            return False, 0.0, ""
        missing = []
        if not state.has_helmet:
            missing.append("hard hat")
        if not state.has_vest:
            missing.append("hi-vis vest")
        if missing:
            conf = state.worker_confidence
            items = " and ".join(missing)
            return True, conf, (
                f"Worker #{state.worker_id} is near machinery "
                f"without {items}."
            )
        return False, 0.0, ""

    @staticmethod
    def _check_posture_anomaly(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        """POST-001: Detect fall-risk posture via bbox aspect ratio."""
        if not state.in_height_zone:
            return False, 0.0, ""
        if state.bbox_aspect_ratio > POSTURE_ASPECT_RATIO_THRESHOLD:
            conf = state.worker_confidence * 0.7  # Reduce confidence for heuristic
            return True, conf, (
                f"Worker #{state.worker_id} shows a horizontal posture "
                f"(aspect ratio {state.bbox_aspect_ratio:.2f}) "
                f"suggesting potential fall risk at height."
            )
        return False, 0.0, ""

    @staticmethod
    def _check_danger_zone_entry(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        """SCENE-001: Unauthorized entry into danger zone without PPE."""
        if not state.in_danger_zone:
            return False, 0.0, ""
        if not state.has_helmet:
            conf = state.worker_confidence
            return True, conf, (
                f"Worker #{state.worker_id} is inside a marked danger "
                f"zone without a hard hat."
            )
        return False, 0.0, ""

    @staticmethod
    def _check_low_visibility(
        state: WorkerPPEState, metadata: dict
    ) -> tuple[bool, float, str]:
        """ENV-001: Low-light conditions escalate vest requirement."""
        brightness = metadata.get("brightness", 128.0)
        if brightness >= LOW_LIGHT_BRIGHTNESS:
            return False, 0.0, ""
        if not state.has_vest:
            conf = state.worker_confidence
            return True, conf, (
                f"Worker #{state.worker_id} is without a hi-vis vest "
                f"in low-visibility conditions "
                f"(brightness={brightness:.0f}/255)."
            )
        return False, 0.0, ""

    # ── Zone Applicability ───────────────────────────────────────────

    @staticmethod
    def _is_zone_applicable(
        rule: SafetyRule, state: WorkerPPEState
    ) -> bool:
        """Check if a rule applies to the worker's current zone.

        Args:
            rule: Safety rule with zone_applicability.
            state: Worker's PPE state with zone flags.

        Returns:
            True if the rule should be evaluated for this worker.
        """
        zone = rule.zone_applicability
        if zone == ZoneType.ALL:
            return True
        if zone == ZoneType.ACTIVE_ZONE:
            return True  # Default: treat entire site as active
        if zone == ZoneType.HEIGHT_ZONE:
            return state.in_height_zone
        if zone == ZoneType.MACHINERY_ZONE:
            return state.near_machinery
        return False

    # ── Scene Verdict ────────────────────────────────────────────────

    @staticmethod
    def _compute_scene_verdict(
        violations: list[Violation],
    ) -> SceneVerdict:
        """Compute overall scene safety verdict.

        Args:
            violations: All detected violations.

        Returns:
            SceneVerdict enum value.
        """
        if not violations:
            return SceneVerdict.SAFE

        has_critical = any(
            v.severity == Severity.CRITICAL.value for v in violations
        )
        has_high = any(
            v.severity == Severity.HIGH.value for v in violations
        )

        if has_critical:
            return SceneVerdict.UNSAFE
        if has_high:
            return SceneVerdict.UNSAFE
        return SceneVerdict.WARNING

    @staticmethod
    def _compute_overall_confidence(
        states: list[WorkerPPEState],
        violations: list[Violation],
    ) -> float:
        """Compute weighted average confidence across all results.

        Args:
            states: All worker states.
            violations: All detected violations.

        Returns:
            Overall confidence score [0, 1].
        """
        confidences = [s.worker_confidence for s in states]
        confidences.extend(v.confidence for v in violations)
        if not confidences:
            return 0.0
        return float(sum(confidences) / len(confidences))

    @staticmethod
    def _check_uncertainty(state: WorkerPPEState) -> Optional[str]:
        """Check if a worker has uncertain detection quality.

        Args:
            state: Worker's PPE state.

        Returns:
            Warning string if uncertain, None otherwise.
        """
        if state.is_occluded:
            return (
                f"Worker #{state.worker_id}: >50% occluded — "
                f"compliance status UNCERTAIN."
            )
        if state.is_distant:
            return (
                f"Worker #{state.worker_id}: too distant "
                f"({state.worker_bbox.height}px) — "
                f"attribute detection skipped."
            )
        if state.worker_confidence < LOW_CONFIDENCE_THRESHOLD:
            return (
                f"Worker #{state.worker_id}: low detection confidence "
                f"({state.worker_confidence:.2f})."
            )
        return None
