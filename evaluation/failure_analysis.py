"""Failure Case Analysis — False Positive / False Negative Report Generator.

Identifies the highest-confidence false positives and false negatives
from model predictions and exports them as a labeled HTML report for
manual review by safety engineers.

Usage:
    python failure_analysis.py \
    --model artifacts/best.pt \
        --images-dir data/processed/images/test \
        --labels-dir data/processed/labels/test \
        --output ./failure_report \
        --top-k 20
"""

import argparse
import base64
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL_PATH = "artifacts/best.pt"

CLASS_NAMES = {0: "worker", 1: "helmet", 2: "no_helmet", 3: "vest",
               4: "no_vest", 5: "harness", 6: "machinery", 7: "danger_zone"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Failure case analysis.")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model weights path.")
    p.add_argument("--images-dir", required=True, help="Test images directory.")
    p.add_argument("--labels-dir", required=True, help="Test labels directory.")
    p.add_argument("--output", default="./failure_report", help="Output dir.")
    p.add_argument("--top-k", type=int, default=20, help="Top-K failures.")
    p.add_argument("--conf", type=float, default=0.25, help="Low conf threshold.")
    p.add_argument("--iou-match", type=float, default=0.5, help="Match IoU.")
    return p.parse_args()


def load_gt_labels(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Load ground-truth labels from YOLO format."""
    labels = []
    if not label_path.exists():
        return labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            labels.append({"class_id": cid, "bbox": (x1, y1, x2, y2)})
    return labels


def compute_iou(b1: tuple, b2: tuple) -> float:
    """Compute IoU between two (x1,y1,x2,y2) bboxes."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def find_failures(
    model: YOLO, images_dir: Path, labels_dir: Path,
    conf: float, iou_match: float, top_k: int
) -> tuple[list[dict], list[dict]]:
    """Find top-K false positives and false negatives."""
    false_positives = []
    false_negatives = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl_path = labels_dir / (img_path.stem + ".txt")
        gt = load_gt_labels(lbl_path, w, h)

        results = model(img, conf=conf, verbose=False)
        preds = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                preds.append({
                    "class_id": int(boxes.cls[i]),
                    "confidence": float(boxes.conf[i]),
                    "bbox": tuple(xyxy),
                })

        gt_matched = [False] * len(gt)
        pred_matched = [False] * len(preds)

        for pi, pred in enumerate(preds):
            for gi, g in enumerate(gt):
                if gt_matched[gi] or pred.get("class_id") != g["class_id"]:
                    continue
                if compute_iou(pred["bbox"], g["bbox"]) >= iou_match:
                    pred_matched[pi] = True
                    gt_matched[gi] = True
                    break

        for pi, pred in enumerate(preds):
            if not pred_matched[pi]:
                false_positives.append({
                    "image": str(img_path), "confidence": pred["confidence"],
                    "class_id": pred["class_id"], "bbox": pred["bbox"],
                    "class_name": CLASS_NAMES.get(pred["class_id"], "?"),
                })

        for gi, g in enumerate(gt):
            if not gt_matched[gi]:
                false_negatives.append({
                    "image": str(img_path), "confidence": 1.0,
                    "class_id": g["class_id"], "bbox": g["bbox"],
                    "class_name": CLASS_NAMES.get(g["class_id"], "?"),
                })

    false_positives.sort(key=lambda x: x["confidence"], reverse=True)
    false_negatives.sort(key=lambda x: x["confidence"], reverse=True)
    return false_positives[:top_k], false_negatives[:top_k]


def image_to_base64(path: str, bbox: tuple, color: tuple) -> str:
    """Load image, draw bbox, return base64 encoded JPEG."""
    img = cv2.imread(path)
    if img is None:
        return ""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("utf-8")


def generate_html_report(
    fps: list[dict], fns: list[dict], output_dir: Path
) -> Path:
    """Generate HTML failure analysis report."""
    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Failure Analysis Report</title>
<style>
body{font-family:sans-serif;background:#1a1a2e;color:#eee;margin:20px;}
h1{color:#e94560;} h2{color:#0f3460;background:#16213e;padding:10px;border-radius:6px;color:#eee;}
.card{display:inline-block;margin:10px;background:#16213e;border-radius:8px;
padding:12px;width:420px;vertical-align:top;}
.card img{width:100%;border-radius:4px;}
.fp{border-left:4px solid #e74c3c;} .fn{border-left:4px solid #f39c12;}
.meta{font-size:12px;color:#aaa;margin-top:6px;}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;
font-weight:bold;margin-right:4px;}
.badge-fp{background:#e74c3c;} .badge-fn{background:#f39c12;color:#333;}
</style></head><body>
<h1>🔍 Failure Case Analysis Report</h1>
"""
    html += f"<p>Top {len(fps)} False Positives | Top {len(fns)} False Negatives</p>"

    html += "<h2>False Positives (model predicted, but no ground truth match)</h2>"
    for i, fp in enumerate(fps):
        b64 = image_to_base64(fp["image"], fp["bbox"], (0, 0, 255))
        html += f"""<div class="card fp">
<img src="data:image/jpeg;base64,{b64}" alt="FP {i+1}">
<div><span class="badge badge-fp">FP</span>
<strong>{fp['class_name']}</strong> (conf: {fp['confidence']:.3f})</div>
<div class="meta">{Path(fp['image']).name}</div></div>"""

    html += "<h2>False Negatives (ground truth exists, but model missed)</h2>"
    for i, fn in enumerate(fns):
        b64 = image_to_base64(fn["image"], fn["bbox"], (0, 165, 255))
        html += f"""<div class="card fn">
<img src="data:image/jpeg;base64,{b64}" alt="FN {i+1}">
<div><span class="badge badge-fn">FN</span>
<strong>{fn['class_name']}</strong></div>
<div class="meta">{Path(fn['image']).name}</div></div>"""

    html += "</body></html>"
    out = output_dir / "failure_analysis.html"
    with open(out, "w") as f:
        f.write(html)
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    fps, fns = find_failures(model, Path(args.images_dir), Path(args.labels_dir),
                             args.conf, args.iou_match, args.top_k)
    print(f"Found {len(fps)} false positives, {len(fns)} false negatives")
    report_path = generate_html_report(fps, fns, out)
    print(f" HTML report saved: {report_path}")


if __name__ == "__main__":
    main()
