# 🏗️ Construction Site Safety Monitoring System

Real-time computer vision pipeline for detecting PPE compliance, hazardous conditions,
and safety rule violations on construction sites using YOLOv8.

## Architecture

```
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Video/Image │───▶│  SafetyDetect │───▶│  ComplianceEngine │───▶│ ReportGen    │
│    Input     │    │  (YOLOv8)     │    │  (10 Rules)       │    │ JSON/CSV/HTML│
└──────────────┘    └───────┬───────┘    └────────┬─────────┘    └──────────────┘
                            │                     │
                    ┌───────▼───────┐    ┌────────▼─────────┐
                    │ PPE-Worker    │    │  SafetyVisualizer │
                    │ Association   │    │  (Annotated Frame)│
                    └───────────────┘    └──────────────────┘

Pipeline Flow:
  1. Frame ingestion + low-light preprocessing (CLAHE)
  2. YOLOv8 detection → workers, PPE items, machinery, danger zones
  3. Spatial PPE-to-worker association (1.3× expanded bbox containment)
  4. Rule engine evaluation (10 rules × each worker)
  5. Compliance report + annotated visualization
```

## Safety Rules

| Rule ID | Rule Name | Severity | Zone | OSHA Ref |
|---------|-----------|----------|------|----------|
| PPE-001 | Hard Hat Required | 🔴 CRITICAL | ALL | 29 CFR 1926.100 |
| PPE-002 | High-Vis Vest Required | 🟠 HIGH | ACTIVE_ZONE | 29 CFR 1926.201 |
| PPE-003 | Safety Harness at Height | 🔴 CRITICAL | HEIGHT_ZONE | 29 CFR 1926.502 |
| PPE-004 | Improper Helmet Usage | 🟠 HIGH | ALL | 29 CFR 1926.100(b) |
| PPE-005 | Improperly Worn Vest | 🟡 MEDIUM | ACTIVE_ZONE | ANSI/ISEA 107-2020 |
| PPE-006 | Complete PPE Ensemble | 🟠 HIGH | ACTIVE_ZONE | 29 CFR 1926.28(a) |
| PROX-001 | Machinery Proximity | 🔴 CRITICAL | MACHINERY_ZONE | 29 CFR 1926.600 |
| POST-001 | Fall Risk Posture | 🟠 HIGH | HEIGHT_ZONE | 29 CFR 1926.501 |
| SCENE-001 | Danger Zone Entry | 🔴 CRITICAL | ALL | 29 CFR 1926.200 |
| ENV-001 | Low Visibility Conditions | 🔴 CRITICAL | ALL | 29 CFR 1926.56 |

## Detection Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `worker` | Any person on site |
| 1 | `helmet` | Hard hat worn on head |
| 2 | `no_helmet` | Head visible without helmet |
| 3 | `vest` | Hi-vis vest on torso |
| 4 | `no_vest` | Torso without hi-vis vest |
| 5 | `harness` | Full-body safety harness |
| 6 | `machinery` | Heavy equipment |
| 7 | `danger_zone` | Marked hazardous area |

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

Collect or download a construction safety dataset with the 8 classes above.
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
    ├── test_compliance.py           # All 10 rule tests
    └── test_reporter.py             # Report format tests
```

## Known Limitations

1. **No true pose estimation**: Fall-risk detection (POST-001) uses bbox aspect ratio as a proxy heuristic. A dedicated pose model (YOLOv8-pose, MediaPipe) would be far more accurate for posture anomaly detection.

2. **Attribute classification requires secondary model**: Detecting whether a helmet is worn *properly* (PPE-004) or a vest is *fastened* (PPE-005) requires the optional MobileNetV3 classifier head which is not included in the base YOLO detection.

3. **No temporal tracking in image mode**: Single-frame inference cannot track worker IDs across time. Video mode would benefit from ByteTrack or BoT-SORT integration for persistent worker tracking and temporal violation smoothing.

4. **Distance/depth estimation is approximate**: The system cannot measure true physical distance to machinery (PROX-001) from a single monocular camera. Proximity is estimated from bounding box center distances normalized by frame diagonal — this is a rough approximation.

5. **Zone detection is semi-manual**: Height zones and danger zones rely on either manual polygon configuration per camera or detection of visual cues (scaffolding, barriers). The system cannot determine absolute height from monocular images without additional calibration.

6. **Night/low-light performance degrades**: While CLAHE preprocessing improves low-light detection, accuracy drops significantly below ~40 lux. Infrared or thermal cameras would be needed for reliable night-shift monitoring.

7. **Class imbalance in real-world data**: `harness` and `danger_zone` classes are typically underrepresented in available datasets, leading to lower per-class AP for these categories.

8. **No hardhat color/type classification**: The system detects helmet presence but does not classify helmet type (standard, full-brim, bump cap) or color coding used for role identification on many sites.

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
