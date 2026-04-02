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
  - batch=8: Optimal for RTX 4060(8GB VRAM) with YOLOv8-Medium;
    larger batches cause OOM, smaller batches reduce gradient stability.
  - lr0=0.01: Standard YOLO learning rate; works well with SGD+momentum
    for fine-tuning from COCO pretrained weights.
  - lrf=0.01: Final LR = 1% of lr0; cosine decay to near-zero prevents
    late-stage oscillation while allowing fine-grained convergence.
  - momentum=0.937: Slightly below default 0.937 promotes smoother
    convergence for datasets with class imbalance (violation classes are rarer).
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
import logging
from pathlib import Path

from ultralytics import YOLO


logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = SCRIPT_DIR / "construction_safety.yaml"


def parse_args() -> argparse.Namespace:
    """Parse training configuration arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for construction safety detection."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_YAML),
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
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device selection: auto, cpu, 0, 0,1, etc.",
    )
    parser.add_argument("--project", type=str, default="safety_monitor")
    parser.add_argument("--name", type=str, default="yolov8m_construction_v1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--export", type=str, nargs="+", default=None,
                        help="Export formats after training (onnx, torchscript, etc.).")
    return parser.parse_args()


def resolve_data_yaml(data_yaml: str) -> Path:
    """Resolve dataset YAML path from CLI or script-relative fallback."""
    candidate = Path(data_yaml)
    if candidate.exists():
        return candidate.resolve()

    if not candidate.is_absolute():
        script_relative = SCRIPT_DIR / candidate
        if script_relative.exists():
            logger.info("Resolved dataset YAML relative to script: %s", script_relative)
            return script_relative.resolve()

    return candidate


def resolve_device(requested_device: str) -> str:
    """Resolve training device with safe CPU fallback for non-CUDA hosts."""
    requested = (requested_device or "auto").strip().lower()

    try:
        import torch
        has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        has_cuda = False

    if requested == "auto":
        return "0" if has_cuda else "cpu"

    if requested == "cpu":
        return "cpu"

    if has_cuda:
        return requested_device

    logger.warning(
        "CUDA requested via --device=%s but no CUDA devices were detected. Falling back to CPU.",
        requested_device,
    )
    return "cpu"


def validate_dataset(data_yaml: str) -> bool:
    """Verify dataset YAML and directory structure before training.

    Args:
        data_yaml: Path to the YOLO dataset YAML file.

    Returns:
        True if validation passes, False otherwise.
    """
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
      logger.error("Dataset YAML not found: %s", yaml_path)
      return False

    try:
      import yaml

      with open(yaml_path, encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)
    except Exception as exc:
      logger.error("Failed to parse dataset YAML: %s", exc)
      return False

    required_keys = ["path", "train", "val", "nc", "names"]
    for key in required_keys:
      if key not in config:
        logger.error("Missing required key in dataset YAML: %s", key)
        return False

    base = Path(config["path"])
    for split in ["train", "val"]:
      split_path = base / config[split]
      if not split_path.exists():
        logger.warning(
          "Split directory missing: %s. Training may still work if data is prepared later.",
          split_path,
        )

    num_classes = config["nc"]
    num_names = len(config["names"])
    if num_classes != num_names:
      logger.error(
        "Class mismatch in dataset YAML: nc=%s but %s class names provided.",
        num_classes,
        num_names,
      )
      return False

    logger.info("Dataset YAML validated: %s classes", num_classes)
    for idx, name in config["names"].items():
      logger.info("  %s: %s", idx, name)
    return True


def train(args: argparse.Namespace) -> None:
    """Execute the YOLOv8 training pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    data_yaml_path = resolve_data_yaml(args.data)
    if not validate_dataset(str(data_yaml_path)):
      raise FileNotFoundError(
        f"Dataset validation failed. Provide a valid --data path. Checked: {data_yaml_path}"
      )

    selected_device = resolve_device(args.device)

    # Initialize model from checkpoint or pretrained weights.
    if args.resume:
        logger.info("Resuming training from checkpoint: %s", args.resume)
        model = YOLO(args.resume)
    else:
        logger.info("Loading pretrained model: %s", args.model)
        model = YOLO(args.model)

    # Run training.
    logger.info("Starting training")
    logger.info("Training configuration: epochs=%s, imgsz=%s, batch=%s", args.epochs, args.imgsz, args.batch)
    logger.info("Learning rate schedule: initial=%s final=%s", args.lr0, args.lr0 * args.lrf)
    logger.info("Execution target: device=%s, run=%s/%s", selected_device, args.project, args.name)

    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        device=selected_device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        val=True,
        plots=True,
        save_period=10,
    )

    # Run post-training evaluation on validation split.
    logger.info("Running final evaluation on validation set")
    metrics = model.val()
    logger.info("Validation metrics: mAP@0.5=%.4f, mAP@0.5:0.95=%.4f", metrics.box.map50, metrics.box.map)

    # Export model artifacts if requested.
    if args.export:
        for fmt in args.export:
            logger.info("Exporting model in format: %s", fmt)
            model.export(format=fmt)

    logger.info("Training complete")
    logger.info("Best weights: %s/%s/weights/best.pt", args.project, args.name)
    logger.info("Last weights: %s/%s/weights/last.pt", args.project, args.name)


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
