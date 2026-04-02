# 🚧 SiteSafe: Construction Safety Monitor

![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

**SiteSafe** is a real-time computer vision pipeline designed to automatically monitor construction site conditions and determine whether a given situation is safe or unsafe. Built to address the critical need for automated safety compliance, this system detects workers, identifies Personal Protective Equipment (PPE), and rigorously evaluates complex safety rules in real-time.

---

## 🎯 The Core Question: Is this situation safe or unsafe?

SiteSafe answers this by breaking the problem into four continuous stages:
1. **Worker Detection:** Locating individuals (`Person` class) within the frame.
2. **PPE Recognition:** Detecting 10 distinct classes of PPE presence and absence (helmets, vests, gloves, etc.).
3. **Compliance Check:** Spatially associating PPE to workers and checking a decoupled JSON rule engine.
4. **Violation Flagging:** Emitting real-time visual alerts and generating structured compliance reports.

---

## 📚 Dataset Documentation & Curation

*Note: Please update the exact dataset numbers based on your final custom data distribution.*

To ensure robustness across varied real-world environments, this model was trained on a **custom dataset** extending a baseline 11-class Construction-PPE schema. 

- **Varied Environments:** Includes scaffolding, indoor warehouses, and open-air active zones.
- **Lighting Conditions:** Augmented and curated to include overcast shadows and harsh sunlight.
- **Classes (11 Total):** 
  - `Person`
  - **Present:** `helmet`, `gloves`, `vest`, `boots`, `goggles`
  - **Missing Signals:** `no_helmet`, `no_gloves`, `no_boots`, `no_goggle`, `none`
- **Annotation Strategy:** Bounding boxes carefully drawn around both the workers and individual PPE items to enable spatial association during inference.

---

## 🛡️ Defining Safety Violations (Rule Engine)

Instead of hardcoding safety logic into the model, SiteSafe utilizes a decoupled, zone-aware JSON rule engine (`rules/safety_rules.json`). This allows safety officers to configure rules without retraining the model.

### Active Rules:
1. **PPE-001 (Helmet Required):** Every `Person` must have an associated `helmet` and lack a `no_helmet` signal. *Severity: CRITICAL*.
2. **PPE-002 (Gloves Required):** Requires `gloves` overlapping with the worker. *Severity: HIGH*.
3. **PPE-003 (High-Visibility Vest):** Required in active working zones to prevent vehicle-pedestrian accidents. *Severity: HIGH*.
4. **PPE-004 (Safety Boots):** Checks for proper footwear (`boots`). *Severity: HIGH*.
5. **PPE-005 (Goggles Required):** Eye protection enforcement. *Severity: HIGH*.
6. **PPE-006 (Complete Ensemble):** Context-aware rule requiring the full 5-piece PPE set for highly hazardous tasks. *Severity: CRITICAL*.

*A violation(Active Rule ID) is flagged when a worker's bounding box lacks the required PPE bounding box or explicitly contains a "missing PPE" signal (e.g., `no_helmet`).*

---

## 🧠 System Architecture & Design Decisions

- **Architecture Choice (YOLOv8):** Selected for its industry-leading balance of real-time inference speed (essential for live video feeds) and detection accuracy.
- **Low-Light Preprocessing:** Integrates optional CLAHE (Contrast Limited Adaptive Histogram Equalization) to handle shadows and poor indoor artificial lighting.
- **Spatial Association Engine:** Uses Intersection over Union (IoU) and bounding-box geometry to map isolated PPE detections to the correct worker, preventing false positives when compliant and non-compliant workers stand next to each other.
- **Reporting UX:** Generates per-frame JSON outputs, shifting summaries, and a persistent CSV log (`reports/compliance_log.csv`) for analytics dashboards.

---

## 🚀 Installation & Quick Start

### Prerequisites
- Python >= 3.13
- (Optional but recommended) CUDA-capable GPU for real-time video inference.

### Setup
```bash
git clone <your-repo-link>
cd SiteSafe
pip install -r requirements.txt
```

### Running Inference
**Image Inference:**
```bash
python -m demo.run_image --image data/raw/images/test/image1.txt --no-display
```

**Video Inference:**
```bash
python -m demo.run_video --source data/raw/video.mp4 --output result.mp4 --save-frame-json
```

---

## 🏋️ Training & Evaluation

To reproduce the model training using the custom YAML configuration:

```bash
python training/train.py --data training/construction_safety.yaml --epochs 100 --device auto
```

Evaluate the model against your test set to calculate Precision, Recall, and mAP:
```bash
python evaluation/evaluate.py --data training/construction_safety.yaml
```

*Note: For detailed failure analysis (false positives/negatives), run `python evaluation/failure_analysis.py`.*

---

## 📊 Honest Evaluation & Known Limitations

**Where the model excels:**
- Real-time performance suitable for multi-camera site deployments.
- Accurately distinguishing between standard PPE presence (helmets, vests) in well-lit environments.
- High configurability thanks to the JSON rule engine.

**Where the model struggles (Known Limitations):**
1. **Severe Occlusion:** If a worker's lower half is completely hidden behind scaffolding, the model may fail to detect `boots`, triggering a false positive violation.
2. **Extreme Low Light:** Despite CLAHE preprocessing, very dark scenes can degrade confidence scores, necessitating manual review.
3. **Temporal Tracking:** The current pipeline evaluates compliance on a per-frame basis. It does not yet include a multi-object tracker (e.g., DeepSORT) to persist worker identities across frames, which could smooth out flickering detections.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
