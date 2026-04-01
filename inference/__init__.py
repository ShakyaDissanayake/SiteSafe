"""Construction Site Safety Monitoring System — Inference Package.

Provides real-time computer vision inference for detecting PPE compliance,
hazardous conditions, and safety rule violations on construction sites.

Modules:
    detector:    YOLOv8-based object detection for workers and PPE items.
    compliance:  Rule engine that evaluates detections against the safety ruleset.
    reporter:    Structured compliance report generation.
    visualizer:  Annotated frame rendering with bounding boxes and HUD overlay.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Safety rule severity levels, ordered by urgency."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ZoneType(str, Enum):
    """Construction site zone classifications."""
    ALL = "ALL"
    ACTIVE_ZONE = "ACTIVE_ZONE"
    HEIGHT_ZONE = "HEIGHT_ZONE"
    MACHINERY_ZONE = "MACHINERY_ZONE"


class SceneVerdict(str, Enum):
    """Overall scene compliance verdict."""
    SAFE = "SAFE"
    WARNING = "WARNING"
    UNSAFE = "UNSAFE"


class DetectionClass(str, Enum):
    """Detection classes aligned with the YOLO training config."""
    WORKER = "worker"
    HELMET = "helmet"
    NO_HELMET = "no_helmet"
    VEST = "vest"
    NO_VEST = "no_vest"
    HARNESS = "harness"
    MACHINERY = "machinery"
    DANGER_ZONE = "danger_zone"


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates (x1, y1, x2, y2)."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of the bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Area of the bounding box in pixels²."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center point (cx, cy) of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def aspect_ratio(self) -> float:
        """Width-to-height aspect ratio."""
        return self.width / max(self.height, 1)

    def expanded(self, factor: float) -> "BBox":
        """Return a new BBox expanded by `factor` on each side.

        Args:
            factor: Fraction to expand (e.g. 0.3 = 30% expansion each side).

        Returns:
            New expanded BBox (clamped to non-negative coordinates).
        """
        dx = int(self.width * factor)
        dy = int(self.height * factor)
        return BBox(
            x1=max(0, self.x1 - dx),
            y1=max(0, self.y1 - dy),
            x2=self.x2 + dx,
            y2=self.y2 + dy,
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point (x, y) lies within this bounding box."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def iou(self, other: "BBox") -> float:
        """Compute Intersection-over-Union with another BBox."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area
        return inter_area / max(union_area, 1e-6)

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return bounding box as a plain tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Detection:
    """Single object detection from the YOLO model."""
    class_name: str
    class_id: int
    confidence: float
    bbox: BBox
    track_id: Optional[int] = None


@dataclass
class WorkerPPEState:
    """PPE compliance state for a single detected worker."""
    worker_id: int
    worker_bbox: BBox
    worker_confidence: float
    has_helmet: bool = False
    has_vest: bool = False
    has_harness: bool = False
    helmet_confidence: float = 0.0
    vest_confidence: float = 0.0
    harness_confidence: float = 0.0
    helmet_proper: Optional[bool] = None
    vest_proper: Optional[bool] = None
    near_machinery: bool = False
    in_height_zone: bool = False
    in_danger_zone: bool = False
    is_occluded: bool = False
    is_distant: bool = False
    bbox_aspect_ratio: float = 0.0
    associated_detections: list[Detection] = field(default_factory=list)


@dataclass
class SafetyRule:
    """A single safety compliance rule loaded from the JSON ruleset."""
    rule_id: str
    rule_name: str
    description: str
    detection_targets: list[str]
    severity: Severity
    zone_applicability: ZoneType
    suggested_action: str
    osha_reference: str = ""


@dataclass
class Violation:
    """A detected safety rule violation for a specific worker."""
    worker_id: int
    rule_id: str
    rule_name: str
    severity: str
    description: str
    confidence: float
    bbox: tuple[int, int, int, int]
    suggested_action: str


@dataclass
class ComplianceReport:
    """Structured compliance report for a single frame or image."""
    frame_id: str
    timestamp: str
    scene_verdict: Literal["SAFE", "UNSAFE", "WARNING"]
    overall_confidence: float
    worker_count: int
    compliant_workers: int
    violation_count: int
    violations: list[Violation]
    annotated_frame: Optional[np.ndarray] = None
    scene_brightness: float = 128.0
    low_confidence_flags: list[str] = field(default_factory=list)


@dataclass
class DetectionResult:
    """Raw detection output from the YOLO model."""
    workers: list[Detection] = field(default_factory=list)
    ppe_items: list[Detection] = field(default_factory=list)
    machinery: list[Detection] = field(default_factory=list)
    danger_zones: list[Detection] = field(default_factory=list)
    raw_detections: list[Detection] = field(default_factory=list)
    frame_brightness: float = 128.0
    preprocessing_applied: list[str] = field(default_factory=list)
