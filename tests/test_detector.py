"""Unit tests for the SafetyDetector module.

Tests BBox utilities and PPE-to-worker association logic without requiring
a loaded model (uses mock detections).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import BBox, Detection, DetectionClass, WorkerPPEState

try:
    from inference.detector import SafetyDetector
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    SafetyDetector = None


# ── BBox Tests ───────────────────────────────────────────────────────────

class TestBBox:
    def test_width_height(self):
        b = BBox(10, 20, 110, 220)
        assert b.width == 100
        assert b.height == 200

    def test_area(self):
        b = BBox(0, 0, 100, 50)
        assert b.area == 5000

    def test_center(self):
        b = BBox(100, 200, 300, 400)
        assert b.center == (200.0, 300.0)

    def test_aspect_ratio(self):
        b = BBox(0, 0, 200, 100)
        assert b.aspect_ratio == 2.0

    def test_contains_point_inside(self):
        b = BBox(10, 10, 100, 100)
        assert b.contains_point(50, 50) is True

    def test_contains_point_outside(self):
        b = BBox(10, 10, 100, 100)
        assert b.contains_point(5, 5) is False

    def test_contains_point_edge(self):
        b = BBox(10, 10, 100, 100)
        assert b.contains_point(10, 10) is True

    def test_expanded(self):
        b = BBox(100, 100, 200, 200)
        e = b.expanded(0.5)
        assert e.x1 == 50
        assert e.y1 == 50
        assert e.x2 == 250
        assert e.y2 == 250

    def test_expanded_no_negative(self):
        b = BBox(5, 5, 15, 15)
        e = b.expanded(1.0)
        assert e.x1 == 0
        assert e.y1 == 0

    def test_iou_identical(self):
        b = BBox(0, 0, 100, 100)
        assert abs(b.iou(b) - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        b1 = BBox(0, 0, 50, 50)
        b2 = BBox(100, 100, 200, 200)
        assert b1.iou(b2) == 0.0

    def test_iou_partial(self):
        b1 = BBox(0, 0, 100, 100)
        b2 = BBox(50, 50, 150, 150)
        iou = b1.iou(b2)
        assert 0.1 < iou < 0.2  # ~14.3% overlap

    def test_as_tuple(self):
        b = BBox(1, 2, 3, 4)
        assert b.as_tuple() == (1, 2, 3, 4)


# ── PPE Association Tests ────────────────────────────────────────────────

def make_detection(cls_name: str, cls_id: int, conf: float,
                   x1: int, y1: int, x2: int, y2: int) -> Detection:
    return Detection(cls_name, cls_id, conf, BBox(x1, y1, x2, y2))


@pytest.mark.skipif(not HAS_ULTRALYTICS, reason="ultralytics not installed")
class TestPPEAssociation:
    """Test PPE-to-worker spatial association (no model required)."""

    def _get_detector_stub(self):
        """Create a minimal detector-like object for testing association."""
        class DetectorStub:
            associate_ppe_to_workers = SafetyDetector.associate_ppe_to_workers
            _build_worker_state = SafetyDetector._build_worker_state
            _assign_ppe_items = SafetyDetector._assign_ppe_items
            _check_machinery_proximity = SafetyDetector._check_machinery_proximity
            _check_danger_zones = SafetyDetector._check_danger_zones
            _assess_visibility = SafetyDetector._assess_visibility
        return DetectorStub()

    def test_helmet_association(self):
        stub = self._get_detector_stub()
        worker = make_detection("worker", 0, 0.9, 100, 100, 300, 500)
        helmet = make_detection("helmet", 1, 0.85, 150, 100, 250, 160)
        states = stub.associate_ppe_to_workers(
            [worker], [helmet], [], [], (640, 640))
        assert len(states) == 1
        assert states[0].has_helmet is True

    def test_no_helmet_association(self):
        stub = self._get_detector_stub()
        worker = make_detection("worker", 0, 0.9, 100, 100, 300, 500)
        no_hat = make_detection("no_helmet", 2, 0.8, 150, 100, 250, 160)
        states = stub.associate_ppe_to_workers(
            [worker], [no_hat], [], [], (640, 640))
        assert states[0].has_helmet is False

    def test_ppe_outside_worker_not_associated(self):
        stub = self._get_detector_stub()
        worker = make_detection("worker", 0, 0.9, 100, 100, 200, 300)
        helmet_far = make_detection("helmet", 1, 0.85, 500, 500, 550, 550)
        states = stub.associate_ppe_to_workers(
            [worker], [helmet_far], [], [], (640, 640))
        assert states[0].has_helmet is False

    def test_vest_association(self):
        stub = self._get_detector_stub()
        worker = make_detection("worker", 0, 0.9, 100, 100, 300, 500)
        vest = make_detection("vest", 3, 0.8, 120, 200, 280, 400)
        states = stub.associate_ppe_to_workers(
            [worker], [vest], [], [], (640, 640))
        assert states[0].has_vest is True

    def test_multiple_workers(self):
        stub = self._get_detector_stub()
        w1 = make_detection("worker", 0, 0.9, 10, 10, 150, 400)
        w2 = make_detection("worker", 0, 0.85, 400, 10, 550, 400)
        h1 = make_detection("helmet", 1, 0.8, 50, 10, 120, 60)
        states = stub.associate_ppe_to_workers(
            [w1, w2], [h1], [], [], (640, 640))
        assert states[0].has_helmet is True   # helmet near worker 1
        assert states[1].has_helmet is False  # no helmet near worker 2

    def test_distant_worker_flagged(self):
        stub = self._get_detector_stub()
        # Worker bbox only 25px tall (below 30px threshold)
        w = make_detection("worker", 0, 0.5, 100, 100, 140, 125)
        states = stub.associate_ppe_to_workers([w], [], [], [], (640, 640))
        assert states[0].is_distant is True

    def test_danger_zone_containment(self):
        stub = self._get_detector_stub()
        worker = make_detection("worker", 0, 0.9, 200, 200, 300, 400)
        dz = make_detection("danger_zone", 7, 0.7, 100, 100, 500, 500)
        states = stub.associate_ppe_to_workers(
            [worker], [], [], [dz], (640, 640))
        assert states[0].in_danger_zone is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
