# 🏗️ Construction Site Safety Monitoring System

Real-time computer vision pipeline for detecting PPE compliance
and safety rule violations on construction sites using YOLOv8.

## Architecture

```
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Video/Image │───▶│  SafetyDetect │───▶│  ComplianceEngine │───▶│ ReportGen    │
│    Input     │    │  (YOLOv8)     │    │  (8 PPE Rules)    │    │ JSON/CSV/HTML│
└──────────────┘    └───────┬───────┘    └────────┬─────────┘    └──────────────┘
                            │                     │
                    ┌───────▼───────┐    ┌────────▼─────────┐
                    │ PPE-Worker    │    │  SafetyVisualizer │
                    │ Association   │    │  (Annotated Frame)│
                    └───────────────┘    └──────────────────┘

Pipeline Flow:
  1. Frame ingestion + low-light preprocessing (CLAHE)
  2. YOLOv8 detection → Person + PPE and no-PPE classes
  3. Spatial PPE-to-worker association (1.3× expanded bbox containment)
  4. Rule engine evaluation (dataset-aligned PPE compliance rules)
  5. Compliance report + annotated visualization
```

## Safety Rules

| Rule ID | Rule Name | Severity | Zone | OSHA Ref |
|---------|-----------|----------|------|----------|
| PPE-001 | Helmet Required | 🔴 CRITICAL | ALL | 29 CFR 1926.100 |
| PPE-002 | Gloves Required | 🟠 HIGH | ALL | - |
| PPE-003 | High-Vis Vest Required | 🟠 HIGH | ALL | 29 CFR 1926.201 |
| PPE-004 | Safety Boots Required | 🟠 HIGH | ALL | - |
| PPE-005 | Goggles Required | 🟠 HIGH | ALL | - |
| PPE-006 | Complete PPE Ensemble | 🔴 CRITICAL | ACTIVE_ZONE | 29 CFR 1926.28(a) |
| PPE-007 | None-Class PPE Violation | 🔴 CRITICAL | ALL | - |
| ENV-001 | Low Visibility Vest Escalation | 🔴 CRITICAL | ALL | 29 CFR 1926.56 |

## Detection Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `helmet` | Helmet detected |
| 1 | `gloves` | Gloves detected |
| 2 | `vest` | High-visibility vest detected |
| 3 | `boots` | Safety boots detected |
| 4 | `goggles` | Eye protection detected |
| 5 | `none` | Explicit no-PPE class |
| 6 | `Person` | Worker/person class |
| 7 | `no_helmet` | Missing helmet label |
| 8 | `no_goggle` | Missing goggles label |
| 9 | `no_gloves` | Missing gloves label |
| 10 | `no_boots` | Missing boots label |

## Prerequisites

- Python 3.10+
- CUDA 11.7+ (GPU inference)
- 4GB+ VRAM (YOLOv8-Medium)

## Installation

```bash
git clone https://github.com/your-org/construction-safety-monitor.git
cd construction-safety-monitor
pip install -r requirements.txt
```

## Quick Start

**Inference on a single image (3 commands):**

```bash
pip install -r requirements.txt
# 📌 Download or provide your trained model weights as best.pt
python -m demo.run_image --model best.pt --image test_photo.jpg --no-display
```

**Video inference:**

```bash
python -m demo.run_video --model best.pt --source video.mp4 --output result.mp4
```

## Training Reproduction

### 1. Prepare Dataset

Use the Construction-PPE dataset schema with the 11 classes above.
Recommended sources:

| Dataset | Images | Format | URL |
|---------|--------|--------|-----|
| SHWD (Safety Helmet Wearing Dataset) | ~7,500 | VOC | [GitHub](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) |
| Roboflow Construction Safety | ~2,800 | YOLO | [Roboflow](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) |
| CHV (Construction Hazard & Violation) | ~1,200 | COCO | [Papers with Code](https://paperswithcode.com/dataset/chv) |
| Hard Hat Workers Dataset | ~5,000 | VOC | [Kaggle](https://www.kaggle.com/andrewmvd/hard-hat-detection) |
| Pictor-v3 PPE Detection | ~2,500 | YOLO | [GitHub](https://github.com/ciber-lab/pictor-ppe) |

### 2. Analyze Dataset

```bash
python data/prepare_dataset.py --data-dir /path/to/dataset --output-dir ./dataset_stats
```

### 3. Train

```bash
python training/train.py --data training/construction_safety.yaml --epochs 100 --device 0
```

Or use the Colab notebook: `notebooks/Construction_Safety_Training.ipynb`

### 4. Evaluate

```bash
python evaluation/evaluate.py --model safety_monitor/*/weights/best.pt --data training/construction_safety.yaml
```

## Evaluation Targets

| Metric | Target | Description |
|--------|--------|-------------|
| mAP@0.5 | ≥ 0.72 | Mean Average Precision at IoU 0.5 |
| VDR | ≥ 0.85 | Violation Detection Rate (TP / total actual) |
| CMR | ≤ 0.08 | Critical Miss Rate (missed critical / total critical) |
| FAR | ≤ 0.15 | False Alarm Rate (FP / compliant scenes) |
| FPS | ≥ 15 | Inference speed on T4 GPU |

## Project Structure

```
construction_safety_monitor/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                          # Raw downloaded datasets
│   ├── processed/                    # Processed YOLO-format data
│   └── prepare_dataset.py           # Dataset statistics generator
├── training/
│   ├── construction_safety.yaml     # YOLO dataset config (8 classes)
│   ├── train.py                     # Training script with validation
│   └── augmentations.py             # Albumentations pipeline
├── inference/
│   ├── __init__.py                  # Data models & enumerations
│   ├── detector.py                  # YOLOv8 detector + PPE association
│   ├── compliance.py                # Safety rule engine (10 rules)
│   ├── reporter.py                  # JSON/CSV/HTML report generator
│   └── visualizer.py                # Annotated frame rendering
├── rules/
│   ├── safety_rules.json            # Machine-readable ruleset
│   └── zone_configs/
│       └── default_zones.json       # Zone configuration template
├── evaluation/
│   ├── evaluate.py                  # Detection + safety metrics
│   └── failure_analysis.py          # FP/FN HTML report generator
├── demo/
│   ├── run_image.py                 # Single image inference demo
│   └── run_video.py                 # Video / webcam inference demo
├── notebooks/
│   └── Construction_Safety_Training.ipynb  # Full Colab notebook
└── tests/
    ├── test_detector.py             # BBox + association tests
    ├── test_compliance.py           # PPE rule tests
    └── test_reporter.py             # Report format tests
```

## Known Limitations

1. **No temporal tracking in image mode**: Single-frame inference cannot keep worker IDs across time. Video mode would benefit from ByteTrack or BoT-SORT.

2. **Low-light uncertainty remains**: CLAHE helps, but very dark frames still degrade PPE attribute reliability.

3. **Class imbalance exists**: Rare classes (for example `no_boots`) have fewer samples and may produce unstable precision/recall.

4. **No fine-grained PPE quality checks**: The current model checks PPE presence/absence, not whether equipment is worn correctly or fastened.

## Future Work Roadmap

- [ ] Integrate YOLOv8-pose for keypoint-based posture analysis
- [ ] Add ByteTrack for multi-frame worker tracking and ID persistence
- [ ] Deploy as edge inference service (ONNX/TensorRT on Jetson)
- [ ] Build real-time dashboard with WebSocket streaming
- [ ] Add audio alert system for critical violations
- [ ] Multi-camera fusion for cross-view worker tracking
- [ ] Automated incident report generation with timestamp clips
- [ ] Fine-grained glove and boot detection (additional PPE classes)
- [ ] Integration with site access control systems

## License

MIT License — See [LICENSE](LICENSE) for details.

---

**⚠️ Disclaimer**: This system is designed as an *assistive* safety monitoring tool. It does not replace human safety officers, proper safety training, or regulatory compliance processes. All critical safety decisions must involve qualified human oversight.
