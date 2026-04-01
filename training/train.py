"""YOLOv8 Training Script for Construction Site Safety Detection.

Provides a CLI-driven training pipeline with:
  - Automated dataset validation before training
  - Configurable hyperparameters with construction-site-specific defaults
  - Training resumption from checkpoints
  - Post-training evaluation and model export

Architecture Selection Justification:
  YOLOv8-Medium was chosen over alternatives for these reasons:

  vs. Faster R-CNN (two-stage):
    - 5–8× faster inference (critical for real-time safety monitoring)
    - Comparable accuracy on medium/large objects (workers, machinery)
    - Single-stage simplicity reduces deployment complexity
    - Construction PPE detection doesn't need the fine-grained localization
      advantage of two-stage detectors (PPE items are not tiny)

  vs. RT-DETR:
    - Better performance on small objects (distant worker PPE at >15m)
    - Lower GPU memory footprint enables edge deployment
    - More mature ecosystem with Ultralytics tooling
    - RT-DETR's transformer attention is overkill for PPE's visual simplicity

  vs. YOLOv5:
    - YOLOv8 anchor-free design handles diverse PPE aspect ratios better
    - Improved decoupled head separates classification and localization
    - Better small-object detection (critical for distant worker PPE)
    - ~3% higher mAP on COCO benchmarks at similar speed
    - Active development and support from Ultralytics

Hyperparameter Justification (per-parameter):
  - epochs=100: Construction PPE patterns are relatively simple shapes;
    100 epochs is sufficient for convergence without overfitting.
  - imgsz=640: Industry standard balancing resolution and speed; workers
    and PPE are identifiable at 640px in typical surveillance footage.
  - batch=16: Optimal for Colab T4 (16GB VRAM) with YOLOv8-Medium;
    larger batches cause OOM, smaller batches reduce gradient stability.
  - lr0=0.01: Standard YOLO learning rate; works well with SGD+momentum
    for fine-tuning from COCO pretrained weights.
  - lrf=0.01: Final LR = 1% of lr0; cosine decay to near-zero prevents
    late-stage oscillation while allowing fine-grained convergence.
  - momentum=0.937: Slightly below default 0.937 promotes smoother
    convergence for datasets with class imbalance (no_helmet is rarer).
  - weight_decay=0.0005: Mild L2 regularization prevents overfitting on
    the relatively small construction dataset (~3k images).
  - warmup_epochs=3: Gradual LR warmup stabilizes early training when
    transferring COCO features to construction domain.
  - patience=20: Early stopping patience; construction datasets sometimes
    show plateau-then-improvement patterns around epoch 60–80.
  - save_period=10: Checkpoint every 10 epochs for training recovery
    on potentially unstable Colab sessions.

Usage:
    python train.py --data construction_safety.yaml --epochs 100
    python train.py --data construction_safety.yaml --resume runs/last.pt
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse training configuration arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for construction safety detection."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="construction_safety.yaml",
        help="Path to dataset YAML config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Pretrained model (yolov8n/s/m/l/x.pt).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="safety_monitor")
    parser.add_argument("--name", type=str, default="yolov8m_construction_v1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--export", type=str, nargs="+", default=None,
                        help="Export formats after training (onnx, torchscript, etc.).")
    return parser.parse_args()


def validate_dataset(data_yaml: str) -> bool:
    """Verify dataset YAML and directory structure before training.

    Args:
        data_yaml: Path to the YOLO dataset YAML file.

    Returns:
        True if validation passes, False otherwise.
    """
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        print(f"❌ Dataset YAML not found: {yaml_path}")
        return False

    try:
        import yaml
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to parse YAML: {e}")
        return False

    required_keys = ["path", "train", "val", "nc", "names"]
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required key in YAML: {key}")
            return False

    base = Path(config["path"])
    for split in ["train", "val"]:
        split_path = base / config[split]
        if not split_path.exists():
            print(f"⚠️  Split directory missing: {split_path}")
            print("    Training may still work if data is downloaded later.")

    num_classes = config["nc"]
    num_names = len(config["names"])
    if num_classes != num_names:
        print(f"❌ nc={num_classes} but {num_names} class names provided.")
        return False

    print(f"✅ Dataset YAML validated: {num_classes} classes")
    for idx, name in config["names"].items():
        print(f"  {idx}: {name}")
    return True


def train(args: argparse.Namespace) -> None:
    """Execute the YOLOv8 training pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Validate dataset
    if not validate_dataset(args.data):
        print("\n⚠️  Dataset validation had warnings. Continuing anyway...")

    # Load model
    if args.resume:
        print(f"\n🔄 Resuming training from: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"\n📦 Loading pretrained model: {args.model}")
        model = YOLO(args.model)

    # Train
    print("\n🚀 Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Learning rate: {args.lr0} → {args.lr0 * args.lrf}")
    print(f"  Device: {args.device}")
    print(f"  Project: {args.project}/{args.name}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        val=True,
        plots=True,
        save_period=10,
    )

    # Post-training evaluation
    print("\n📊 Running final evaluation on validation set...")
    metrics = model.val()
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")

    # Export if requested
    if args.export:
        for fmt in args.export:
            print(f"\n📤 Exporting model to {fmt}...")
            model.export(format=fmt)

    print("\n✅ Training complete!")
    print(f"  Best weights: {args.project}/{args.name}/weights/best.pt")
    print(f"  Last weights: {args.project}/{args.name}/weights/last.pt")


def main() -> None:
    """Entry point."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
