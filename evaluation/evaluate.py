"""Comprehensive Evaluation Framework for Construction Safety Detection.

Reports both standard detection metrics and safety-system-specific metrics:

Detection Metrics (per class):
  - Precision, Recall, F1 at IoU=0.5
  - mAP@0.5 and mAP@0.5:0.95
  - Per-class AP table

Safety System Metrics:
  - Violation Detection Rate (VDR): TP violations / total actual violations
  - False Alarm Rate (FAR): FP violations / total compliant scenes
  - Critical Miss Rate (CMR): missed CRITICAL violations / total critical
  - Inference speed (FPS on target hardware)

Target Benchmarks:
  - mAP@0.5 >= 0.72
  - VDR >= 0.85
  - CMR <= 0.08
  - FAR <= 0.15
  - Inference >= 15 FPS on T4 GPU

Usage:
    python evaluate.py --model artifacts/best.pt --data training/construction_safety.yaml --device 0
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ── Target Benchmarks ────────────────────────────────────────────────────
BENCHMARKS = {
    "mAP@0.5": 0.72,
    "VDR": 0.85,
    "CMR_max": 0.08,
    "FAR_max": 0.15,
    "FPS_min": 15.0,
}

DEFAULT_MODEL_PATH = "artifacts/best.pt"


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""
    p = argparse.ArgumentParser(description="Evaluate construction safety model.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to model weights.")
    p.add_argument("--data", type=str, required=True, help="Dataset YAML path.")
    p.add_argument("--device", type=str, default="0", help="Device (0, cpu).")
    p.add_argument("--output", type=str, default="./eval_results", help="Output dir.")
    p.add_argument("--imgsz", type=int, default=640, help="Image size.")
    p.add_argument("--conf", type=float, default=0.45, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold.")
    p.add_argument("--fps-frames", type=int, default=100, help="Frames for FPS test.")
    return p.parse_args()


def run_detection_evaluation(model: YOLO, args: argparse.Namespace) -> dict:
    """Run standard YOLO validation and collect detection metrics.

    Args:
        model: Loaded YOLO model.
        args: CLI arguments.

    Returns:
        Dict with mAP values and per-class AP.
    """
    print("\n📊 Running detection evaluation...")
    metrics = model.val(data=args.data, imgsz=args.imgsz,
                        conf=args.conf, iou=args.iou,
                        device=args.device, plots=True)

    class_names = model.names
    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class": {},
    }

    for i, name in class_names.items():
        if i < len(metrics.box.ap50):
            results["per_class"][name] = {
                "AP50": float(metrics.box.ap50[i]),
                "AP50_95": float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0.0,
            }

    return results


def measure_inference_speed(
    model: YOLO, args: argparse.Namespace
) -> float:
    """Measure average inference FPS on synthetic frames.

    Args:
        model: Loaded YOLO model.
        args: CLI arguments.

    Returns:
        Average FPS.
    """
    print(f"\n⏱️  Measuring inference speed ({args.fps_frames} frames)...")
    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        model(dummy, verbose=False, device=args.device)

    start = time.perf_counter()
    for _ in range(args.fps_frames):
        model(dummy, verbose=False, device=args.device)
    elapsed = time.perf_counter() - start

    fps = args.fps_frames / elapsed
    print(f"  Average FPS: {fps:.1f}")
    return fps


def compute_safety_metrics(
    tp_violations: int,
    fp_violations: int,
    fn_violations: int,
    total_compliant_scenes: int,
    missed_critical: int,
    total_critical: int,
) -> dict:
    """Compute safety-system-specific metrics.

    Args:
        tp_violations: True positive violation detections.
        fp_violations: False positive violation detections.
        fn_violations: False negative (missed) violations.
        total_compliant_scenes: Scenes with no ground-truth violations.
        missed_critical: Missed CRITICAL severity violations.
        total_critical: Total CRITICAL severity violations.

    Returns:
        Dict with VDR, FAR, CMR values.
    """
    vdr = tp_violations / max(tp_violations + fn_violations, 1)
    far = fp_violations / max(total_compliant_scenes, 1)
    cmr = missed_critical / max(total_critical, 1)
    return {"VDR": vdr, "FAR": far, "CMR": cmr}


def format_results_table(
    det_metrics: dict, safety_metrics: dict, fps: float
) -> str:
    """Create a formatted results table with benchmark comparison.

    Args:
        det_metrics: Detection evaluation results.
        safety_metrics: Safety system metrics.
        fps: Inference speed.

    Returns:
        Formatted string table.
    """
    lines = [
        "=" * 65,
        "  EVALUATION RESULTS",
        "=" * 65,
        "",
        "  DETECTION METRICS",
        f"  {'Metric':<25} {'Value':>10} {'Benchmark':>12} {'Status':>8}",
        f"  {'─' * 25} {'─' * 10} {'─' * 12} {'─' * 8}",
    ]

    map50 = det_metrics["mAP50"]
    map_status = "✅" if map50 >= BENCHMARKS["mAP@0.5"] else "❌"
    lines.append(f"  {'mAP@0.5':<25} {map50:>10.4f} {'>= 0.72':>12} {map_status:>8}")
    lines.append(f"  {'mAP@0.5:0.95':<25} {det_metrics['mAP50_95']:>10.4f}")

    lines.append(f"\n  PER-CLASS AP@0.5:")
    for cls, vals in det_metrics.get("per_class", {}).items():
        lines.append(f"    {cls:<20} {vals['AP50']:>8.4f}")

    lines.append(f"\n  SAFETY SYSTEM METRICS")
    lines.append(f"  {'Metric':<25} {'Value':>10} {'Benchmark':>12} {'Status':>8}")
    lines.append(f"  {'─' * 25} {'─' * 10} {'─' * 12} {'─' * 8}")

    vdr = safety_metrics.get("VDR", 0)
    vdr_s = "✅" if vdr >= BENCHMARKS["VDR"] else "❌"
    lines.append(f"  {'VDR':<25} {vdr:>10.4f} {'>= 0.85':>12} {vdr_s:>8}")

    cmr = safety_metrics.get("CMR", 0)
    cmr_s = "✅" if cmr <= BENCHMARKS["CMR_max"] else "❌"
    lines.append(f"  {'CMR':<25} {cmr:>10.4f} {'<= 0.08':>12} {cmr_s:>8}")

    far = safety_metrics.get("FAR", 0)
    far_s = "✅" if far <= BENCHMARKS["FAR_max"] else "❌"
    lines.append(f"  {'FAR':<25} {far:>10.4f} {'<= 0.15':>12} {far_s:>8}")

    fps_s = "✅" if fps >= BENCHMARKS["FPS_min"] else "❌"
    lines.append(f"  {'Inference FPS':<25} {fps:>10.1f} {'>= 15':>12} {fps_s:>8}")

    lines.append("=" * 65)
    return "\n".join(lines)


def main() -> None:
    """Run the full evaluation pipeline."""
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    det_metrics = run_detection_evaluation(model, args)
    fps = measure_inference_speed(model, args)

    # Safety metrics need ground-truth violation annotations.
    # Provide placeholder values; replace with real annotation comparison.
    print("\n📌 REQUIRES: Ground-truth violation annotations for VDR/FAR/CMR.")
    print("   Using placeholder values. Replace with your test set evaluation.")
    safety_metrics = compute_safety_metrics(
        tp_violations=0, fp_violations=0, fn_violations=0,
        total_compliant_scenes=1, missed_critical=0, total_critical=1,
    )

    table = format_results_table(det_metrics, safety_metrics, fps)
    print(table)

    # Save results
    results = {"detection": det_metrics, "safety": safety_metrics, "fps": fps}
    with open(out / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out / "eval_results.txt", "w") as f:
        f.write(table)

    print(f"\n✅ Results saved to {out}/")


if __name__ == "__main__":
    main()
