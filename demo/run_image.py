"""Run safety inference on a single image.

Usage:
    python -m demo.run_image --image photo.jpg
    python -m demo.run_image --model artifacts/best.pt --image photo.jpg --output result.jpg --rules rules/safety_rules.json
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.detector import SafetyDetector
from inference.compliance import ComplianceEngine
from inference.reporter import ReportGenerator
from inference.visualizer import SafetyVisualizer


DEFAULT_MODEL_PATH = "artifacts/best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run safety detection on an image.")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to YOLO weights.")
    p.add_argument("--image", required=True, help="Input image path.")
    p.add_argument("--output", default=None, help="Output annotated image path.")
    p.add_argument("--rules", default="rules/safety_rules.json", help="Safety rules JSON.")
    p.add_argument("--conf", type=float, default=0.45, help="Confidence threshold.")
    p.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/0).")
    p.add_argument("--no-display", action="store_true", help="Don't show window.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load image
    frame = cv2.imread(args.image)
    if frame is None:
        print(f" Cannot read image: {args.image}")
        sys.exit(1)
    print(f"📷 Loaded image: {args.image} ({frame.shape[1]}x{frame.shape[0]})")

    # Initialize pipeline
    detector = SafetyDetector(args.model, confidence_threshold=args.conf, device=args.device)
    engine = ComplianceEngine.from_json(args.rules)
    reporter = ReportGenerator()
    visualizer = SafetyVisualizer()

    # Run detection
    print("🔍 Running detection...")
    result = detector.detect(frame)
    print(
        f"  Found: {len(result.workers)} persons and {len(result.ppe_items)} PPE/no-PPE detections"
    )

    # Associate PPE to workers
    worker_states = detector.associate_ppe_to_workers(
        result.workers, result.ppe_items,
        result.machinery, result.danger_zones,
        frame.shape[:2],
    )

    # Evaluate compliance
    metadata = {
        "frame_id": Path(args.image).stem,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "brightness": result.frame_brightness,
    }
    report = engine.evaluate(worker_states, metadata)

    # Print report
    reporter.print_summary(report)

    # Visualize
    annotated = visualizer.draw_frame(
        frame, worker_states, report,
        result.danger_zones, result.machinery,
    )
    report.annotated_frame = annotated

    # Save output
    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f" Saved: {args.output}")
    else:
        out = f"{Path(args.image).stem}_annotated.jpg"
        cv2.imwrite(out, annotated)
        print(f" Saved: {out}")

    # Save JSON report
    reporter.save_json(report, f"{Path(args.image).stem}_report.json")

    # Display
    if not args.no_display:
        cv2.imshow("Safety Analysis", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
