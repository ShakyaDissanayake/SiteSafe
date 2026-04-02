"""Run safety inference on a video file or webcam stream.

Usage:
    python -m demo.run_video --model best.pt --source video.mp4
    python -m demo.run_video --model best.pt --source 0          # webcam
    python -m demo.run_video --model best.pt --source video.mp4 --output result.mp4
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.detector import SafetyDetector
from inference.compliance import ComplianceEngine
from inference.reporter import ReportGenerator, ShiftAggregator
from inference.visualizer import SafetyVisualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run safety detection on video.")
    p.add_argument("--model", required=True, help="Path to YOLO weights.")
    p.add_argument("--source", required=True, help="Video path or camera index.")
    p.add_argument("--output", default=None, help="Output video path.")
    p.add_argument("--rules", default="rules/safety_rules.json")
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device. Use 'cpu' for stability or explicit CUDA device like '0'.",
    )
    p.add_argument("--max-frames", type=int, default=0, help="Max frames (0=all).")
    p.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame.")
    p.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory for generated report artifacts.",
    )
    p.add_argument(
        "--save-frame-json",
        action="store_true",
        help="Save per-frame JSON reports in addition to CSV and shift summary.",
    )
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def resolve_device(requested_device: str) -> str:
    """Resolve runtime device with CPU-first default behavior.

    Args:
        requested_device: User-provided device string.

    Returns:
        Safe device string for YOLO inference.
    """
    device = (requested_device or "cpu").strip().lower()
    if device in {"", "auto"}:
        return "cpu"

    if device == "cpu":
        return "cpu"

    if not torch.cuda.is_available():
        print(
            f"WARNING: CUDA device '{requested_device}' requested but CUDA is unavailable. Falling back to CPU."
        )
        return "cpu"

    return requested_device


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {args.source}")
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source: {args.source} ({w}x{h} @ {fps_in:.1f} FPS, {total} frames)")
    print(f"Device: {device}")

    # Init pipeline
    detector = SafetyDetector(args.model, confidence_threshold=args.conf, device=device)
    engine = ComplianceEngine.from_json(args.rules)
    reporter = ReportGenerator(output_dir=args.reports_dir)
    visualizer = SafetyVisualizer()
    aggregator = ShiftAggregator()

    # Output writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_in, (w, h))

    frame_idx = 0
    processed = 0
    start_time = time.perf_counter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if args.max_frames and frame_idx > args.max_frames:
                break
            if frame_idx % args.skip_frames != 0:
                if writer:
                    writer.write(frame)
                continue

            result = detector.detect(frame)
            states = detector.associate_ppe_to_workers(
                result.workers, result.ppe_items,
                result.machinery, result.danger_zones, frame.shape[:2])

            metadata = {"frame_id": f"frame_{frame_idx:06d}",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "brightness": result.frame_brightness}
            report = engine.evaluate(states, metadata)
            aggregator.add_report(report)
            reporter.append_csv(report)
            if args.save_frame_json:
                reporter.save_json(report, f"{report.frame_id}.json")

            annotated = visualizer.draw_frame(
                frame, states, report, result.danger_zones, result.machinery)

            if writer:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("Safety Monitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            if processed % 50 == 0:
                elapsed = time.perf_counter() - start_time
                fps = processed / elapsed
                print(f"  Frame {frame_idx}/{total} | FPS: {fps:.1f} | "
                      f"Violations: {report.violation_count}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    print(f"\nProcessed {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f} FPS)")

    # Print and save shift summary
    summary = aggregator.get_summary()
    print("\nShift Summary:")
    print(f"  Frames: {summary.get('total_frames', 0)}")
    print(f"  Unsafe frames: {summary.get('unsafe_frames', 0)} "
          f"({summary.get('unsafe_rate', 0):.1%})")
    print(f"  Total violations: {summary.get('total_violations', 0)}")

    summary_path = Path(args.reports_dir) / "shift_summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"Report summary saved: {summary_path}")
    print(f"Report CSV saved: {Path(args.reports_dir) / 'compliance_log.csv'}")

    if args.output:
        print(f"Output video saved: {args.output}")


if __name__ == "__main__":
    main()
