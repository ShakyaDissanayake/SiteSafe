"""Run safety inference on a video file or webcam stream.

Usage:
    python -m demo.run_video --model best.pt --source video.mp4
    python -m demo.run_video --model best.pt --source 0          # webcam
    python -m demo.run_video --model best.pt --source video.mp4 --output result.mp4
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

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
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-frames", type=int, default=0, help="Max frames (0=all).")
    p.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame.")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open: {args.source}")
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Source: {args.source} ({w}x{h} @ {fps_in:.1f} FPS, {total} frames)")

    # Init pipeline
    detector = SafetyDetector(args.model, confidence_threshold=args.conf, device=args.device)
    engine = ComplianceEngine.from_json(args.rules)
    reporter = ReportGenerator()
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
    print(f"\n✅ Processed {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f} FPS)")

    # Print shift summary
    summary = aggregator.get_summary()
    print(f"\n📊 Shift Summary:")
    print(f"  Frames: {summary.get('total_frames', 0)}")
    print(f"  Unsafe frames: {summary.get('unsafe_frames', 0)} "
          f"({summary.get('unsafe_rate', 0):.1%})")
    print(f"  Total violations: {summary.get('total_violations', 0)}")

    if args.output:
        print(f"💾 Saved: {args.output}")


if __name__ == "__main__":
    main()
