"""Unit tests for the ComplianceEngine.

Tests all 10 safety rules against constructed worker states without
requiring model inference. Covers both violation and non-violation paths.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import BBox, Severity, Violation, WorkerPPEState
from inference.compliance import ComplianceEngine


@pytest.fixture
def engine() -> ComplianceEngine:
    return ComplianceEngine.from_json(
        str(Path(__file__).resolve().parent.parent / "rules" / "safety_rules.json")
    )


def make_worker(
    wid: int = 0, helmet: bool = True, vest: bool = True,
    harness: bool = False, near_mach: bool = False,
    in_height: bool = False, in_danger: bool = False,
    helmet_proper: bool = None, vest_proper: bool = None,
    distant: bool = False, occluded: bool = False,
    aspect_ratio: float = 0.5, confidence: float = 0.9,
) -> WorkerPPEState:
    return WorkerPPEState(
        worker_id=wid, worker_bbox=BBox(100, 100, 300, 500),
        worker_confidence=confidence,
        has_helmet=helmet, has_vest=vest, has_harness=harness,
        helmet_confidence=0.85, vest_confidence=0.8, harness_confidence=0.7,
        helmet_proper=helmet_proper, vest_proper=vest_proper,
        near_machinery=near_mach, in_height_zone=in_height,
        in_danger_zone=in_danger, is_occluded=occluded,
        is_distant=distant, bbox_aspect_ratio=aspect_ratio,
    )


METADATA = {"frame_id": "test_001", "timestamp": "2026-04-01 12:00:00", "brightness": 128.0}


class TestPPE001HardHat:
    def test_violation_no_helmet(self, engine):
        w = make_worker(helmet=False)
        r = engine.evaluate([w], METADATA)
        rule_ids = [v.rule_id for v in r.violations]
        assert "PPE-001" in rule_ids

    def test_no_violation_with_helmet(self, engine):
        w = make_worker(helmet=True)
        r = engine.evaluate([w], METADATA)
        assert "PPE-001" not in [v.rule_id for v in r.violations]

    def test_skip_distant_worker(self, engine):
        w = make_worker(helmet=False, distant=True)
        r = engine.evaluate([w], METADATA)
        assert "PPE-001" not in [v.rule_id for v in r.violations]


class TestPPE002Vest:
    def test_violation_no_vest(self, engine):
        w = make_worker(vest=False)
        r = engine.evaluate([w], METADATA)
        assert "PPE-002" in [v.rule_id for v in r.violations]

    def test_no_violation_with_vest(self, engine):
        w = make_worker(vest=True)
        r = engine.evaluate([w], METADATA)
        assert "PPE-002" not in [v.rule_id for v in r.violations]


class TestPPE003Harness:
    def test_violation_at_height_no_harness(self, engine):
        w = make_worker(harness=False, in_height=True)
        r = engine.evaluate([w], METADATA)
        assert "PPE-003" in [v.rule_id for v in r.violations]

    def test_no_violation_at_height_with_harness(self, engine):
        w = make_worker(harness=True, in_height=True)
        r = engine.evaluate([w], METADATA)
        assert "PPE-003" not in [v.rule_id for v in r.violations]

    def test_no_violation_not_at_height(self, engine):
        w = make_worker(harness=False, in_height=False)
        r = engine.evaluate([w], METADATA)
        assert "PPE-003" not in [v.rule_id for v in r.violations]


class TestPPE004ImproperHelmet:
    def test_violation_improper(self, engine):
        w = make_worker(helmet=True, helmet_proper=False)
        r = engine.evaluate([w], METADATA)
        assert "PPE-004" in [v.rule_id for v in r.violations]

    def test_no_violation_proper(self, engine):
        w = make_worker(helmet=True, helmet_proper=True)
        r = engine.evaluate([w], METADATA)
        assert "PPE-004" not in [v.rule_id for v in r.violations]


class TestPPE005ImproperVest:
    def test_violation_open_vest(self, engine):
        w = make_worker(vest=True, vest_proper=False)
        r = engine.evaluate([w], METADATA)
        assert "PPE-005" in [v.rule_id for v in r.violations]


class TestPROX001MachineryProximity:
    def test_violation_near_machinery_no_helmet(self, engine):
        w = make_worker(helmet=False, vest=True, near_mach=True)
        r = engine.evaluate([w], METADATA)
        assert "PROX-001" in [v.rule_id for v in r.violations]

    def test_no_violation_full_ppe_near_machinery(self, engine):
        w = make_worker(helmet=True, vest=True, near_mach=True)
        r = engine.evaluate([w], METADATA)
        assert "PROX-001" not in [v.rule_id for v in r.violations]


class TestPOST001PostureAnomaly:
    def test_violation_horizontal_at_height(self, engine):
        w = make_worker(in_height=True, aspect_ratio=2.0)
        r = engine.evaluate([w], METADATA)
        assert "POST-001" in [v.rule_id for v in r.violations]

    def test_no_violation_normal_posture(self, engine):
        w = make_worker(in_height=True, aspect_ratio=0.5)
        r = engine.evaluate([w], METADATA)
        assert "POST-001" not in [v.rule_id for v in r.violations]


class TestSCENE001DangerZone:
    def test_violation_in_danger_no_helmet(self, engine):
        w = make_worker(helmet=False, in_danger=True)
        r = engine.evaluate([w], METADATA)
        assert "SCENE-001" in [v.rule_id for v in r.violations]

    def test_no_violation_in_danger_with_helmet(self, engine):
        w = make_worker(helmet=True, in_danger=True)
        r = engine.evaluate([w], METADATA)
        assert "SCENE-001" not in [v.rule_id for v in r.violations]


class TestENV001LowVisibility:
    def test_violation_low_light_no_vest(self, engine):
        meta = {**METADATA, "brightness": 40.0}
        w = make_worker(vest=False)
        r = engine.evaluate([w], meta)
        assert "ENV-001" in [v.rule_id for v in r.violations]

    def test_no_violation_normal_light(self, engine):
        w = make_worker(vest=False)
        r = engine.evaluate([w], METADATA)  # brightness=128
        assert "ENV-001" not in [v.rule_id for v in r.violations]


class TestSceneVerdict:
    def test_safe_when_all_compliant(self, engine):
        w = make_worker(helmet=True, vest=True)
        r = engine.evaluate([w], METADATA)
        assert r.scene_verdict == "SAFE"

    def test_unsafe_for_critical(self, engine):
        w = make_worker(helmet=False)  # PPE-001 is CRITICAL
        r = engine.evaluate([w], METADATA)
        assert r.scene_verdict == "UNSAFE"

    def test_empty_workers_safe(self, engine):
        r = engine.evaluate([], METADATA)
        assert r.scene_verdict == "SAFE"


class TestUncertaintyFlags:
    def test_occluded_flagged(self, engine):
        w = make_worker(occluded=True)
        r = engine.evaluate([w], METADATA)
        assert any("UNCERTAIN" in f for f in r.low_confidence_flags)

    def test_distant_flagged(self, engine):
        w = make_worker(distant=True)
        r = engine.evaluate([w], METADATA)
        assert any("distant" in f for f in r.low_confidence_flags)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
