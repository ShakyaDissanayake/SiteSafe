"""Unit tests for SafetyDetector PPE mapping with Construction-PPE classes."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import BBox, Detection

try:
    from inference.detector import SafetyDetector

    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    SafetyDetector = None


def det(cls_name: str, cls_id: int, conf: float, x1: int, y1: int, x2: int, y2: int) -> Detection:
    return Detection(cls_name, cls_id, conf, BBox(x1, y1, x2, y2))


@pytest.mark.skipif(not HAS_ULTRALYTICS, reason="ultralytics not installed")
class TestPPEAssociation:
    def _stub(self):
        class DetectorStub:
            associate_ppe_to_workers = SafetyDetector.associate_ppe_to_workers
            _build_worker_state = SafetyDetector._build_worker_state
            _assign_ppe_items = SafetyDetector._assign_ppe_items
            _check_machinery_proximity = SafetyDetector._check_machinery_proximity
            _check_danger_zones = SafetyDetector._check_danger_zones
            _assess_visibility = SafetyDetector._assess_visibility

        return DetectorStub()

    def test_person_with_helmet_and_vest(self):
        stub = self._stub()
        worker = det("Person", 6, 0.92, 100, 100, 320, 520)
        helmet = det("helmet", 0, 0.86, 140, 120, 210, 180)
        vest = det("vest", 2, 0.88, 130, 240, 270, 430)

        states = stub.associate_ppe_to_workers([worker], [helmet, vest], [], [], (640, 640))
        assert states[0].has_helmet is True
        assert states[0].has_vest is True

    def test_negative_signals_override_presence(self):
        stub = self._stub()
        worker = det("Person", 6, 0.9, 100, 100, 320, 520)
        gloves = det("gloves", 1, 0.8, 120, 300, 170, 360)
        no_gloves = det("no_gloves", 9, 0.85, 125, 305, 175, 365)

        states = stub.associate_ppe_to_workers([worker], [gloves, no_gloves], [], [], (640, 640))
        assert states[0].has_gloves is False

    def test_none_class_flag(self):
        stub = self._stub()
        worker = det("Person", 6, 0.9, 100, 100, 320, 520)
        none_cls = det("none", 5, 0.8, 140, 210, 260, 420)

        states = stub.associate_ppe_to_workers([worker], [none_cls], [], [], (640, 640))
        assert states[0].none_class_detected is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
