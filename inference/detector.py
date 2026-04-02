"""Core Safety Detection Module.

Provides the SafetyDetector class that wraps YOLOv8 inference with
construction-site-specific preprocessing (CLAHE for low-light),
detection partitioning, and spatial PPE-to-worker association.

Typical usage:
    detector = SafetyDetector("best.pt", confidence_threshold=0.45)
    result = detector.detect(frame)
    worker_states = detector.associate_ppe_to_workers(
        result.workers, result.ppe_items
    )
"""

from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from inference import (
    BBox,
    Detection,
    DetectionClass,
    DetectionResult,
    WorkerPPEState,
)



LOW_LIGHT_THRESHOLD = 60       # Mean brightness below this triggers CLAHE
DISTANT_WORKER_HEIGHT_PX = 30  # Workers shorter than this skip attribute checks
OCCLUSION_THRESHOLD = 0.50     # IoU overlap ratio suggesting occlusion
PPE_EXPANSION_FACTOR = 0.30    # Enlarge worker bbox by 30% for PPE association
MACHINERY_PROXIMITY_RATIO = 0.25  # Max center distance ratio for proximity check

# Class IDs matching construction_safety.yaml
_PERSON_ID = 6
_PPE_CLASS_IDS = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10}


class SafetyDetector:
    """YOLOv8-based construction safety object detector.

    Handles model loading, frame preprocessing, detection inference,
    and spatial PPE-to-worker association.

    Attributes:
        model: Loaded YOLOv8 model instance.
        confidence_threshold: Minimum confidence for valid detections.
        iou_threshold: NMS IoU threshold for overlapping detections.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.50,
        device: str = "auto",
    ) -> None:
        """Initialize the safety detector.

        Args:
            model_path: Path to YOLOv8 weights (.pt file).
            confidence_threshold: Minimum detection confidence.
            iou_threshold: NMS IoU threshold.
            device: Inference device ('auto', 'cpu', '0', 'cuda:0').
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a single frame.

        Applies low-light preprocessing if needed, runs YOLOv8
        inference, and partitions detections by category.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            DetectionResult with categorized detections.
        """
        result = DetectionResult()
        preprocessed, result.preprocessing_applied = self._preprocess(frame)
        result.frame_brightness = self._compute_brightness(frame)

        # Run YOLOv8 inference
        predictions = self.model(
            preprocessed,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        if not predictions or len(predictions) == 0:
            return result

        # Parse detections from YOLO results
        raw = self._parse_predictions(predictions[0])
        result.raw_detections = raw

        # Partition into categories
        for det in raw:
            if det.class_id == _PERSON_ID:
                result.workers.append(det)
            elif det.class_id in _PPE_CLASS_IDS:
                result.ppe_items.append(det)

        return result

    def associate_ppe_to_workers(
        self,
        workers: list[Detection],
        ppe_items: list[Detection],
        machinery: Optional[list[Detection]] = None,
        danger_zones: Optional[list[Detection]] = None,
        frame_shape: Optional[tuple[int, int]] = None,
    ) -> list[WorkerPPEState]:
        """Associate PPE detections with worker bounding boxes.

        Uses spatial containment: a PPE item is assigned to a worker if
        the PPE's centroid falls within the worker's expanded bounding box
        (expanded by PPE_EXPANSION_FACTOR = 1.3x).

        Args:
            workers: List of worker detections.
            ppe_items: List of PPE item detections.
            machinery: Optional list of machinery detections.
            danger_zones: Optional list of danger zone detections.
            frame_shape: Optional (height, width) for proximity calcs.

        Returns:
            List of WorkerPPEState objects with PPE assignments.
        """
        machinery = machinery or []
        danger_zones = danger_zones or []
        states = []

        for idx, worker in enumerate(workers):
            state = self._build_worker_state(idx, worker)
            expanded_bbox = worker.bbox.expanded(PPE_EXPANSION_FACTOR)

            # Associate PPE items
            self._assign_ppe_items(state, expanded_bbox, ppe_items)

            # Check proximity to machinery
            self._check_machinery_proximity(
                state, worker, machinery, frame_shape
            )

            # Check danger zone containment
            self._check_danger_zones(state, worker, danger_zones)

            # Assess occlusion and distance
            self._assess_visibility(state, worker, workers, idx)

            states.append(state)

        return states

   

    def _preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """Apply construction-site-specific preprocessing.

        Args:
            frame: Input BGR frame.

        Returns:
            Tuple of (processed_frame, list_of_applied_operations).
        """
        applied = []
        processed = frame.copy()
        brightness = self._compute_brightness(frame)

        # CLAHE for low-light / night scenes
        if brightness < LOW_LIGHT_THRESHOLD:
            processed = self._apply_clahe(processed)
            applied.append(f"CLAHE (brightness={brightness:.0f})")

        return processed, applied

    @staticmethod
    def _compute_brightness(frame: np.ndarray) -> float:
        """Compute mean frame brightness (0–255).

        Args:
            frame: BGR image.

        Returns:
            Mean brightness value.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def _apply_clahe(frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance low-light images.

        Args:
            frame: BGR image.

        Returns:
            Enhanced BGR image.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _parse_predictions(self, result) -> list[Detection]:
        """Convert YOLO result to Detection objects.

        Args:
            result: Single YOLOv8 result object.

        Returns:
            List of Detection objects.
        """
        detections = []
        boxes = result.boxes
        if boxes is None:
            return detections

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

            track_id = None
            if boxes.id is not None:
                track_id = int(boxes.id[i].cpu().numpy())

            detections.append(Detection(
                class_name=cls_name,
                class_id=cls_id,
                confidence=conf,
                bbox=BBox(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                ),
                track_id=track_id,
            ))

        return detections

    @staticmethod
    def _build_worker_state(
        idx: int, worker: Detection
    ) -> WorkerPPEState:
        """Create initial WorkerPPEState for a worker detection.

        Args:
            idx: Worker index.
            worker: Worker detection object.

        Returns:
            Initialized WorkerPPEState.
        """
        return WorkerPPEState(
            worker_id=worker.track_id if worker.track_id else idx,
            worker_bbox=worker.bbox,
            worker_confidence=worker.confidence,
            bbox_aspect_ratio=worker.bbox.aspect_ratio,
        )

    @staticmethod
    def _assign_ppe_items(
        state: WorkerPPEState,
        expanded_bbox: BBox,
        ppe_items: list[Detection],
    ) -> None:
        """Assign PPE items to a worker based on spatial containment.

        A PPE centroid must fall within the worker's expanded bbox.

        Args:
            state: Worker's PPE state to update.
            expanded_bbox: Worker bbox expanded by PPE_EXPANSION_FACTOR.
            ppe_items: All detected PPE items.
        """
        for ppe in ppe_items:
            cx, cy = ppe.bbox.center
            if not expanded_bbox.contains_point(cx, cy):
                continue

            state.associated_detections.append(ppe)
            cls = ppe.class_name

            if cls == DetectionClass.HELMET.value:
                state.has_helmet = True
                state.helmet_confidence = max(
                    state.helmet_confidence, ppe.confidence
                )
            elif cls == DetectionClass.GLOVES.value:
                state.has_gloves = True
                state.gloves_confidence = max(
                    state.gloves_confidence, ppe.confidence
                )
            elif cls == DetectionClass.VEST.value:
                state.has_vest = True
                state.vest_confidence = max(
                    state.vest_confidence, ppe.confidence
                )
            elif cls == DetectionClass.BOOTS.value:
                state.has_boots = True
                state.boots_confidence = max(
                    state.boots_confidence, ppe.confidence
                )
            elif cls == DetectionClass.GOGGLES.value:
                state.has_goggles = True
                state.goggles_confidence = max(
                    state.goggles_confidence, ppe.confidence
                )
            elif cls == DetectionClass.NONE.value:
                state.none_class_detected = True
            elif cls == DetectionClass.NO_HELMET.value:
                state.has_helmet = False
                state.helmet_confidence = max(
                    state.helmet_confidence, ppe.confidence
                )
            elif cls == DetectionClass.NO_GLOVES.value:
                state.has_gloves = False
                state.gloves_confidence = max(
                    state.gloves_confidence, ppe.confidence
                )
            elif cls == DetectionClass.NO_BOOTS.value:
                state.has_boots = False
                state.boots_confidence = max(
                    state.boots_confidence, ppe.confidence
                )
            elif cls == DetectionClass.NO_GOGGLE.value:
                state.has_goggles = False
                state.goggles_confidence = max(
                    state.goggles_confidence, ppe.confidence
                )

    @staticmethod
    def _check_machinery_proximity(
        state: WorkerPPEState,
        worker: Detection,
        machinery: list[Detection],
        frame_shape: Optional[tuple[int, int]],
    ) -> None:
        """Check if worker is proximate to any machinery.

        Uses Euclidean distance between bbox centers normalized by
        frame diagonal.

        Args:
            state: Worker's PPE state to update.
            worker: Worker detection.
            machinery: List of machinery detections.
            frame_shape: (height, width) of the frame.
        """
        if not machinery or not frame_shape:
            return

        h, w = frame_shape
        diag = np.sqrt(h**2 + w**2)
        wcx, wcy = worker.bbox.center

        for mach in machinery:
            mcx, mcy = mach.bbox.center
            dist = np.sqrt((wcx - mcx) ** 2 + (wcy - mcy) ** 2)
            if dist / diag < MACHINERY_PROXIMITY_RATIO:
                state.near_machinery = True
                break

    @staticmethod
    def _check_danger_zones(
        state: WorkerPPEState,
        worker: Detection,
        danger_zones: list[Detection],
    ) -> None:
        """Check if worker is inside any detected danger zone.

        Args:
            state: Worker's PPE state to update.
            worker: Worker detection.
            danger_zones: List of danger zone detections.
        """
        wcx, wcy = worker.bbox.center
        for zone in danger_zones:
            if zone.bbox.contains_point(wcx, wcy):
                state.in_danger_zone = True
                break

    @staticmethod
    def _assess_visibility(
        state: WorkerPPEState,
        worker: Detection,
        all_workers: list[Detection],
        current_idx: int,
    ) -> None:
        """Assess whether worker is occluded or too distant.

        Args:
            state: Worker's PPE state to update.
            worker: Current worker detection.
            all_workers: All detected workers.
            current_idx: Index of current worker.
        """
        # Distant worker check
        if worker.bbox.height < DISTANT_WORKER_HEIGHT_PX:
            state.is_distant = True

        # Occlusion check — high IoU with another worker
        for i, other in enumerate(all_workers):
            if i == current_idx:
                continue
            if worker.bbox.iou(other.bbox) > OCCLUSION_THRESHOLD:
                state.is_occluded = True
                break
