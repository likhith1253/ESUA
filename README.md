# Explainable Scene Understanding Assistant (ESUA)

**ESUA** is a modular AI system designed to understand scenes, detect objects, identify potential risks (e.g., "cup near laptop"), and provide human-readable explanations in real-time. It runs efficiently on CPU using lightweight models like YOLOv8n.

## 🚀 Project Overview

The project is built in **Phases**, each adding a layer of intelligence to the system:

| Phase | Component | Description |
| :--- | :--- | :--- |
| **Phase 1** | **Object Detection** | Uses YOLOv8 to identify objects (cups, laptops, people) in images. |
| **Phase 2** | **Spatial Understanding** | Calculates distances and relationships between objects (e.g., "A is near B"). |
| **Phase 3** | **Context Reasoning** | Applies common-sense rules to find risks (e.g., Liquids + Electronics = Spill Risk). |
| **Phase 4** | **Explanation Generation** | Converts risk data into clear, natural language warnings. |
| **Phase 6** | **Camera Integration** | The final real-time application that runs the full pipeline on a webcam feed. |

> *Note: Phase 5 was integrated into the refinement of previous phases.*

---

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd ML project -1
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Key libraries: `ultralytics`, `opencv-python`, `numpy`.*

---

## 🏃 Connectivity & Usage

### 1. Run the Real-Time Assistant (Main App)
To start the live webcam feed with risk analysis:
```bash
python ESUA/phase6_camera_integration/camera_runner.py
```
- **Controls**: Press `q` to quit.

### 2. Run High-Accuracy Snapshot Mode
To confirm observations using multi-frame analysis:
```bash
python ESUA/phase6_camera_integration/snapshot_analyzer.py
```
- **Controls**: Press `c` to capture and analyze, `q` to quit.

### 3. Test Individual Phases
You can run specific phases to see how the logic works step-by-step:

- **Detection Demo**:
  ```bash
  python ESUA/phase1_object_detection/detect_image.py
  ```
- **Spatial Logic**:
  ```bash
  python ESUA/phase2_spatial_understanding/spatial_relations.py
  ```
- **Risk Reasoning**:
  ```bash
  python ESUA/phase3_context_reasoning/context_reasoning.py
  ```
- **Explanation Generator**:
  ```bash
  python ESUA/phase4_explanation_generation/explanation_generator.py
  ```

---

## 📂 Directory Structure

```text
ESUA/
├── phase1_object_detection/       # YOLOv8 implementation
├── phase2_spatial_understanding/  # Geometry and distance logic
├── phase3_context_reasoning/      # Risk rules and object categories
├── phase4_explanation_generation/ # Templates for text generation
└── phase6_camera_integration/     # Live monitor & snapshot tools
README.md                          # This file
requirements.txt                   # python dependencies
```

## 🧠 Experimental
- `main.py`: A standalone script for testing BLIP image captioning (separate from the main ESUA pipeline).
