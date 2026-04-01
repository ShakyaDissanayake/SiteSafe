"""Frame Visualization Module.

Draws annotated bounding boxes, compliance labels, severity indicators,
and a HUD overlay on video frames using OpenCV.

Color scheme:
  - Green:  Compliant worker (has all required PPE)
  - Red:    Violation detected (with rule ID labels)
  - Yellow: Low-confidence detection (<0.5)
  - Blue:   Machinery / scene elements
  - Pink:   Danger zone overlay (semi-transparent)

Typical usage:
    viz = SafetyVisualizer()
    annotated = viz.draw_frame(frame, worker_states, report)
"""

from typing import Optional

import cv2
import numpy as np

from inference import (
    ComplianceReport,
    Detection,
    Violation,
    WorkerPPEState,
)


# ── Color Constants (BGR for OpenCV) ─────────────────────────────────────
COLOR_SAFE = (39, 174, 96)        # Green
COLOR_VIOLATION = (60, 76, 231)   # Red
COLOR_LOW_CONF = (0, 215, 255)    # Yellow
COLOR_MACHINERY = (212, 144, 74)  # Blue
COLOR_DANGER = (99, 30, 233)      # Pink
COLOR_TEXT_BG = (30, 30, 30)      # Dark gray
COLOR_WHITE = (255, 255, 255)
COLOR_HUD_BG = (20, 20, 20)

LOW_CONF_THRESHOLD = 0.50
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL = 0.5
FONT_SCALE_HUD = 0.55
FONT_THICKNESS = 1
BBOX_THICKNESS = 2


class SafetyVisualizer:
    """Draws safety compliance annotations on video frames."""

    def __init__(
        self,
        show_confidence: bool = True,
        show_hud: bool = True,
        danger_zone_alpha: float = 0.25,
    ) -> None:
        """Initialize the visualizer.

        Args:
            show_confidence: Show confidence scores on labels.
            show_hud: Show the top-left HUD overlay.
            danger_zone_alpha: Opacity for danger zone overlays.
        """
        self.show_confidence = show_confidence
        self.show_hud = show_hud
        self.danger_zone_alpha = danger_zone_alpha

    def draw_frame(
        self,
        frame: np.ndarray,
        worker_states: list[WorkerPPEState],
        report: ComplianceReport,
        danger_zones: Optional[list[Detection]] = None,
        machinery: Optional[list[Detection]] = None,
    ) -> np.ndarray:
        """Draw all annotations on a frame.

        Args:
            frame: BGR image (H, W, 3).
            worker_states: Worker PPE states from detector.
            report: Compliance report for this frame.
            danger_zones: Optional danger zone detections.
            machinery: Optional machinery detections.

        Returns:
            Annotated BGR frame.
        """
        canvas = frame.copy()

        # Layer 1: Danger zone overlays
        if danger_zones:
            canvas = self._draw_danger_zones(canvas, danger_zones)

        # Layer 2: Machinery bboxes
        if machinery:
            for m in machinery:
                self._draw_bbox(canvas, m.bbox.as_tuple(),
                                "machinery", m.confidence, COLOR_MACHINERY)

        # Layer 3: Worker bboxes with compliance status
        violation_map = self._build_violation_map(report.violations)
        for state in worker_states:
            canvas = self._draw_worker(canvas, state, violation_map)

        # Layer 4: HUD overlay
        if self.show_hud:
            canvas = self._draw_hud(canvas, report)

        return canvas

    def _draw_worker(
        self,
        canvas: np.ndarray,
        state: WorkerPPEState,
        violation_map: dict[int, list[Violation]],
    ) -> np.ndarray:
        """Draw a single worker with compliance annotations.

        Args:
            canvas: Current frame.
            state: Worker's PPE state.
            violation_map: Worker ID to violations mapping.

        Returns:
            Updated canvas.
        """
        bbox = state.worker_bbox.as_tuple()
        violations = violation_map.get(state.worker_id, [])
        is_low_conf = state.worker_confidence < LOW_CONF_THRESHOLD

        # Determine color and label
        if is_low_conf:
            color = COLOR_LOW_CONF
            label = f"W#{state.worker_id} LOW CONF"
        elif violations:
            color = COLOR_VIOLATION
            rule_ids = " ".join(v.rule_id for v in violations)
            label = f"W#{state.worker_id} {rule_ids}"
        else:
            color = COLOR_SAFE
            label = f"W#{state.worker_id} COMPLIANT"

        # Draw bbox
        self._draw_bbox(canvas, bbox, label,
                        state.worker_confidence, color)

        # Draw severity indicator bar
        if violations:
            canvas = self._draw_severity_bar(canvas, bbox, violations)

        return canvas

    def _draw_bbox(
        self,
        canvas: np.ndarray,
        bbox: tuple[int, int, int, int],
        label: str,
        confidence: float,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a bounding box with label.

        Args:
            canvas: Frame to draw on.
            bbox: (x1, y1, x2, y2) coordinates.
            label: Text label for the box.
            confidence: Detection confidence score.
            color: BGR color tuple.
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

        # Build label text
        text = label
        if self.show_confidence:
            text += f" {confidence:.0%}"

        # Label background
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE_LABEL, FONT_THICKNESS)
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(canvas, (x1, label_y - th - 4),
                      (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(canvas, text, (x1 + 2, label_y - 2),
                    FONT, FONT_SCALE_LABEL, COLOR_WHITE, FONT_THICKNESS)

    @staticmethod
    def _draw_severity_bar(
        canvas: np.ndarray,
        bbox: tuple[int, int, int, int],
        violations: list[Violation],
    ) -> np.ndarray:
        """Draw a colored severity indicator bar below the bbox.

        Args:
            canvas: Frame to draw on.
            bbox: Worker bounding box.
            violations: List of violations for this worker.

        Returns:
            Updated canvas.
        """
        sev_colors = {
            "CRITICAL": (60, 76, 231),
            "HIGH": (34, 126, 230),
            "MEDIUM": (0, 215, 255),
            "LOW": (39, 174, 96),
        }
        x1, _, x2, y2 = bbox
        max_sev = "LOW"
        priority = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        for v in violations:
            if priority.get(v.severity, 0) > priority.get(max_sev, 0):
                max_sev = v.severity
        bar_color = sev_colors.get(max_sev, (128, 128, 128))
        cv2.rectangle(canvas, (x1, y2), (x2, y2 + 4), bar_color, -1)
        return canvas

    def _draw_danger_zones(
        self,
        canvas: np.ndarray,
        danger_zones: list[Detection],
    ) -> np.ndarray:
        """Draw semi-transparent danger zone overlays.

        Args:
            canvas: Frame to draw on.
            danger_zones: Danger zone detections.

        Returns:
            Updated canvas with overlays.
        """
        overlay = canvas.copy()
        for dz in danger_zones:
            x1, y1, x2, y2 = dz.bbox.as_tuple()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_DANGER, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_DANGER, 2)
        return cv2.addWeighted(overlay, self.danger_zone_alpha,
                               canvas, 1 - self.danger_zone_alpha, 0)

    @staticmethod
    def _draw_hud(
        canvas: np.ndarray,
        report: ComplianceReport,
    ) -> np.ndarray:
        """Draw the top-left HUD with scene summary.

        Args:
            canvas: Frame to draw on.
            report: Compliance report.

        Returns:
            Updated canvas.
        """
        verdict_colors = {"SAFE": (39, 174, 96), "WARNING": (0, 215, 255),
                          "UNSAFE": (60, 76, 231)}
        lines = [
            f"Verdict: {report.scene_verdict}",
            f"Workers: {report.worker_count}  Violations: {report.violation_count}",
            f"Confidence: {report.overall_confidence:.1%}",
            f"Time: {report.timestamp}",
        ]
        # Draw HUD background
        hud_h = 20 + len(lines) * 22
        hud_w = 320
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (8 + hud_w, 8 + hud_h), COLOR_HUD_BG, -1)
        canvas = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0)
        # Draw text
        vc = verdict_colors.get(report.scene_verdict, COLOR_WHITE)
        for i, line in enumerate(lines):
            color = vc if i == 0 else COLOR_WHITE
            cv2.putText(canvas, line, (16, 28 + i * 22),
                        FONT, FONT_SCALE_HUD, color, FONT_THICKNESS)
        return canvas

    @staticmethod
    def _build_violation_map(
        violations: list[Violation],
    ) -> dict[int, list[Violation]]:
        """Group violations by worker ID."""
        vmap: dict[int, list[Violation]] = {}
        for v in violations:
            vmap.setdefault(v.worker_id, []).append(v)
        return vmap
