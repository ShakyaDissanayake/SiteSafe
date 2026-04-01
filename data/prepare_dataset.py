"""Dataset Preparation & Statistics for Construction Site Safety Monitoring.

Reads a YOLO-format dataset directory and produces comprehensive statistics:
  - Class distribution chart
  - Image resolution histogram
  - Train/val/test split summary
  - Bounding box size distribution
  - Per-class instance counts and annotation density

Usage:
    python prepare_dataset.py --data-dir /path/to/dataset --output-dir ./stats

Expected YOLO directory structure:
    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


# ── Class Mapping (must match construction_safety.yaml) ──────────────────
CLASS_NAMES = {
    0: "worker",
    1: "helmet",
    2: "no_helmet",
    3: "vest",
    4: "no_vest",
    5: "harness",
    6: "machinery",
    7: "danger_zone",
}

# Colors for each class (RGBA)
CLASS_COLORS = {
    0: "#4A90D9",  # worker — blue
    1: "#27AE60",  # helmet — green
    2: "#E74C3C",  # no_helmet — red
    3: "#2ECC71",  # vest — light green
    4: "#E67E22",  # no_vest — orange
    5: "#9B59B6",  # harness — purple
    6: "#F39C12",  # machinery — amber
    7: "#E91E63",  # danger_zone — pink
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate statistics for a YOLO-format construction safety dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory of the YOLO dataset (contains images/ and labels/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dataset_stats",
        help="Directory to save output charts and summary.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="Override default class names (space-separated, in order).",
    )
    return parser.parse_args()


def discover_splits(data_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Discover train/val/test splits in the dataset directory.

    Args:
        data_dir: Root dataset directory.

    Returns:
        Dict mapping split name to (images_dir, labels_dir) tuples.
    """
    splits = {}
    for split_name in ["train", "val", "test"]:
        img_dir = data_dir / "images" / split_name
        lbl_dir = data_dir / "labels" / split_name
        if img_dir.exists() and lbl_dir.exists():
            splits[split_name] = (img_dir, lbl_dir)
    return splits


def parse_yolo_label(label_path: Path) -> list[dict]:
    """Parse a single YOLO-format label file.

    Args:
        label_path: Path to a .txt label file.

    Returns:
        List of annotation dicts with keys: class_id, cx, cy, w, h.
    """
    annotations = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                annotations.append({
                    "class_id": int(parts[0]),
                    "cx": float(parts[1]),
                    "cy": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4]),
                })
    except (ValueError, IOError) as e:
        print(f"  ⚠ Skipping malformed label: {label_path} ({e})")
    return annotations


def get_image_resolution(image_path: Path) -> Optional[tuple[int, int]]:
    """Read image dimensions without loading full pixel data.

    Args:
        image_path: Path to an image file.

    Returns:
        Tuple (width, height) or None if unreadable.
    """
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def collect_statistics(
    splits: dict[str, tuple[Path, Path]],
    class_names: dict[int, str],
) -> dict:
    """Collect comprehensive dataset statistics across all splits.

    Args:
        splits: Dict mapping split name to (images_dir, labels_dir).
        class_names: Mapping of class_id to class name.

    Returns:
        Dict containing all collected statistics.
    """
    stats = {
        "split_counts": {},
        "class_counts": Counter(),
        "class_counts_per_split": defaultdict(Counter),
        "resolutions": [],
        "bbox_widths": [],
        "bbox_heights": [],
        "bbox_areas": [],
        "annotations_per_image": [],
        "images_without_labels": 0,
        "total_images": 0,
        "total_annotations": 0,
    }

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    for split_name, (img_dir, lbl_dir) in splits.items():
        image_files = sorted([
            f for f in img_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        stats["split_counts"][split_name] = len(image_files)
        stats["total_images"] += len(image_files)

        for img_path in image_files:
            # Resolution
            res = get_image_resolution(img_path)
            if res:
                stats["resolutions"].append(res)

            # Labels
            label_path = lbl_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                stats["images_without_labels"] += 1
                stats["annotations_per_image"].append(0)
                continue

            annotations = parse_yolo_label(label_path)
            stats["annotations_per_image"].append(len(annotations))
            stats["total_annotations"] += len(annotations)

            for ann in annotations:
                cid = ann["class_id"]
                stats["class_counts"][cid] += 1
                stats["class_counts_per_split"][split_name][cid] += 1
                stats["bbox_widths"].append(ann["w"])
                stats["bbox_heights"].append(ann["h"])
                stats["bbox_areas"].append(ann["w"] * ann["h"])

    return stats


def plot_class_distribution(
    stats: dict,
    class_names: dict[int, str],
    output_dir: Path,
) -> None:
    """Generate and save class distribution bar chart.

    Args:
        stats: Collected dataset statistics.
        class_names: Mapping of class_id to class name.
        output_dir: Directory to save the chart.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ids = sorted(stats["class_counts"].keys())
    names = [class_names.get(i, f"class_{i}") for i in ids]
    counts = [stats["class_counts"][i] for i in ids]
    colors = [CLASS_COLORS.get(i, "#888888") for i in ids]

    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Class Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Detection Class", fontsize=12)
    ax.set_ylabel("Instance Count", fontsize=12)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close(fig)
    print("  ✓ Saved class_distribution.png")


def plot_resolution_histogram(stats: dict, output_dir: Path) -> None:
    """Generate and save image resolution histogram.

    Args:
        stats: Collected dataset statistics.
        output_dir: Directory to save the chart.
    """
    if not stats["resolutions"]:
        print("  ⚠ No resolution data available.")
        return

    widths = [r[0] for r in stats["resolutions"]]
    heights = [r[1] for r in stats["resolutions"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(widths, bins=30, color="#4A90D9", edgecolor="white", alpha=0.85)
    axes[0].set_title("Image Width Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Count")

    axes[1].hist(heights, bins=30, color="#E67E22", edgecolor="white", alpha=0.85)
    axes[1].set_title("Image Height Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Height (px)")
    axes[1].set_ylabel("Count")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "resolution_histogram.png", dpi=150)
    plt.close(fig)
    print("  ✓ Saved resolution_histogram.png")


def plot_bbox_distribution(stats: dict, output_dir: Path) -> None:
    """Generate and save bounding box size distribution plots.

    Args:
        stats: Collected dataset statistics.
        output_dir: Directory to save the chart.
    """
    if not stats["bbox_widths"]:
        print("  ⚠ No bounding box data available.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(
        stats["bbox_widths"], bins=50,
        color="#2ECC71", edgecolor="white", alpha=0.85,
    )
    axes[0].set_title("BBox Width (Normalized)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Width (0-1)")

    axes[1].hist(
        stats["bbox_heights"], bins=50,
        color="#9B59B6", edgecolor="white", alpha=0.85,
    )
    axes[1].set_title("BBox Height (Normalized)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Height (0-1)")

    axes[2].hist(
        stats["bbox_areas"], bins=50,
        color="#E74C3C", edgecolor="white", alpha=0.85,
    )
    axes[2].set_title("BBox Area (Normalized)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Area (0-1)")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "bbox_distribution.png", dpi=150)
    plt.close(fig)
    print("  ✓ Saved bbox_distribution.png")


def print_summary(stats: dict, class_names: dict[int, str]) -> str:
    """Print and return a formatted text summary of dataset statistics.

    Args:
        stats: Collected dataset statistics.
        class_names: Mapping of class_id to class name.

    Returns:
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 65)
    lines.append("  CONSTRUCTION SAFETY DATASET — STATISTICS SUMMARY")
    lines.append("=" * 65)

    # Split summary
    lines.append("\n📂 Split Summary:")
    lines.append(f"  {'Split':<12} {'Images':>10}")
    lines.append(f"  {'─' * 12} {'─' * 10}")
    for split, count in stats["split_counts"].items():
        lines.append(f"  {split:<12} {count:>10,}")
    lines.append(f"  {'TOTAL':<12} {stats['total_images']:>10,}")

    # Annotation summary
    lines.append(f"\n📊 Annotations: {stats['total_annotations']:,} total")
    ann_per_img = stats["annotations_per_image"]
    if ann_per_img:
        lines.append(f"  Mean per image:  {np.mean(ann_per_img):.1f}")
        lines.append(f"  Median per image: {np.median(ann_per_img):.1f}")
        lines.append(f"  Max per image:   {max(ann_per_img)}")
    lines.append(f"  Images w/o labels: {stats['images_without_labels']}")

    # Class distribution
    lines.append("\n🏷️  Class Distribution:")
    lines.append(f"  {'ID':<4} {'Class':<15} {'Count':>10} {'%':>8}")
    lines.append(f"  {'─' * 4} {'─' * 15} {'─' * 10} {'─' * 8}")
    total = stats["total_annotations"] or 1
    for cid in sorted(stats["class_counts"].keys()):
        count = stats["class_counts"][cid]
        pct = count / total * 100
        name = class_names.get(cid, f"class_{cid}")
        lines.append(f"  {cid:<4} {name:<15} {count:>10,} {pct:>7.1f}%")

    # Resolution summary
    if stats["resolutions"]:
        widths = [r[0] for r in stats["resolutions"]]
        heights = [r[1] for r in stats["resolutions"]]
        lines.append(f"\n📐 Resolution Range:")
        lines.append(
            f"  Width:  {min(widths)}–{max(widths)} px "
            f"(mean {np.mean(widths):.0f})"
        )
        lines.append(
            f"  Height: {min(heights)}–{max(heights)} px "
            f"(mean {np.mean(heights):.0f})"
        )

    # Bbox sizes
    if stats["bbox_areas"]:
        areas = stats["bbox_areas"]
        lines.append(f"\n📏 BBox Area (normalized 0–1):")
        lines.append(f"  Mean: {np.mean(areas):.4f}")
        lines.append(f"  Median: {np.median(areas):.4f}")
        lines.append(
            f"  Small (<0.01): {sum(1 for a in areas if a < 0.01):,} "
            f"({sum(1 for a in areas if a < 0.01) / len(areas) * 100:.1f}%)"
        )
        lines.append(
            f"  Large (>0.10): {sum(1 for a in areas if a > 0.10):,} "
            f"({sum(1 for a in areas if a > 0.10) / len(areas) * 100:.1f}%)"
        )

    lines.append("\n" + "=" * 65)
    summary = "\n".join(lines)
    print(summary)
    return summary


def main() -> None:
    """Entry point: parse args, collect stats, generate outputs."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override class names if provided
    class_names = dict(CLASS_NAMES)
    if args.class_names:
        class_names = {i: name for i, name in enumerate(args.class_names)}

    print(f"\n🔍 Scanning dataset: {data_dir}")
    splits = discover_splits(data_dir)
    if not splits:
        print("❌ No valid splits found. Expected: images/{train,val,test}/")
        sys.exit(1)

    print(f"  Found splits: {', '.join(splits.keys())}")

    print("\n📈 Collecting statistics...")
    stats = collect_statistics(splits, class_names)

    print("\n📊 Generating charts...")
    plot_class_distribution(stats, class_names, output_dir)
    plot_resolution_histogram(stats, output_dir)
    plot_bbox_distribution(stats, output_dir)

    # Print and save text summary
    summary = print_summary(stats, class_names)
    summary_path = output_dir / "dataset_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\n✅ Summary saved to {summary_path}")
    print(f"✅ Charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
