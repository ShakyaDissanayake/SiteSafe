"""Unit tests for dataset-aligned ComplianceEngine PPE rules."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import BBox, WorkerPPEState
from inference.compliance import ComplianceEngine


@pytest.fixture
def engine() -> ComplianceEngine:
    return ComplianceEngine.from_json(
        str(Path(__file__).resolve().parent.parent / "rules" / "safety_rules.json")
    )


def make_worker(
    wid: int = 0,
    helmet: bool = True,
    gloves: bool = True,
    vest: bool = True,
    boots: bool = True,
    goggles: bool = True,
    none_cls: bool = False,
    distant: bool = False,
    occluded: bool = False,
    confidence: float = 0.9,
) -> WorkerPPEState:
    return WorkerPPEState(
        worker_id=wid,
        worker_bbox=BBox(100, 100, 300, 500),
        worker_confidence=confidence,
        has_helmet=helmet,
        has_gloves=gloves,
        has_vest=vest,
        has_boots=boots,
        has_goggles=goggles,
        none_class_detected=none_cls,
        helmet_confidence=0.8,
        gloves_confidence=0.8,
        vest_confidence=0.8,
        boots_confidence=0.8,
        goggles_confidence=0.8,
        is_distant=distant,
        is_occluded=occluded,
    )


META = {
    "frame_id": "test_001",
    "timestamp": "2026-04-01 12:00:00",
    "brightness": 128.0,
}


def rule_ids(report) -> list[str]:
    return [v.rule_id for v in report.violations]


def test_helmet_violation(engine):
    r = engine.evaluate([make_worker(helmet=False)], META)
    assert "PPE-001" in rule_ids(r)


def test_gloves_violation(engine):
    r = engine.evaluate([make_worker(gloves=False)], META)
    assert "PPE-002" in rule_ids(r)


def test_vest_violation(engine):
    r = engine.evaluate([make_worker(vest=False)], META)
    assert "PPE-003" in rule_ids(r)


def test_boots_violation(engine):
    r = engine.evaluate([make_worker(boots=False)], META)
    assert "PPE-004" in rule_ids(r)


def test_goggles_violation(engine):
    r = engine.evaluate([make_worker(goggles=False)], META)
    assert "PPE-005" in rule_ids(r)


def test_full_ppe_violation(engine):
    r = engine.evaluate([make_worker(helmet=False, gloves=False)], META)
    assert "PPE-006" in rule_ids(r)


def test_none_class_violation(engine):
    r = engine.evaluate([make_worker(none_cls=True)], META)
    assert "PPE-007" in rule_ids(r)


def test_low_visibility_escalation(engine):
    meta = {**META, "brightness": 35.0}
    r = engine.evaluate([make_worker(vest=False)], meta)
    assert "ENV-001" in rule_ids(r)


def test_safe_when_all_ppe(engine):
    r = engine.evaluate([make_worker()], META)
    assert r.scene_verdict == "SAFE"


def test_distant_worker_skips_fine_ppe_checks(engine):
    r = engine.evaluate([make_worker(helmet=False, distant=True)], META)
    assert "PPE-001" not in rule_ids(r)


def test_occluded_worker_flagged_uncertain(engine):
    r = engine.evaluate([make_worker(occluded=True)], META)
    assert any("uncertain" in s.lower() for s in r.low_confidence_flags)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
