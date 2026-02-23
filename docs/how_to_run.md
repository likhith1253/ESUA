# How to Run ESUA Locally

This document explains the steps to configure, prepare, and execute ESUA on a local operating system.

## 1. Ensure Dependencies
Make sure you have followed the steps outlined in the [Setup Guide](setup.md) to install `requirements.txt`.

## 2. Connect Your Camera
By default, the `camera_runner.py` and `snapshot_analyzer.py` files hook into OpenCV's `VideoCapture(0)` feed. Ensure your webcam is:
- **Plugged in**
- **Not currently in use** by Zoom, Teams, or another software application.

## 3. Launching the System
Navigate to the root directory and boot up the fully-integrated Phase 6 Analyzer. This encompasses object detection, risk tracking, and spatial understanding logic.

```bash
cd ESUA
python ESUA/phase6_camera_integration/snapshot_analyzer.py
```

## 4. Operational Controls & Analysis
Upon execution, a window titled **"ESUA Live Feed"** will appear on the screen. ESUA leverages a unique buffer-capture system.

### Running Live Proximity Inference
While the live feed is running, the system caches 5 concurrent frames.
- Press `c` on your keyboard to instantly snapshot the 5-frame buffer and execute the deep spatial analysis model on the aggregated data.
- The system will process for several seconds (loading weights, filtering transient ghost-objects, computing bounding box matrices, evaluating risk schemas).
- Once processed, it will superimpose bounding boxes explicitly around items of interest, dropping items filtered out as noise. Textual alert labels will be generated across the UI evaluating proximity threats (e.g., Spill Risk).

### Expected Outputs
The console (stdout terminal) will echo:
1. Model loading confirmation and validation.
2. A textual readout mapping all successfully grouped and confirmed objects.
3. Spatial relationship statements (e.g., *Laptop is near Cup (Distance: 154px)*).
4. Alert warnings triggered by risk rules.

The GUI will freeze on the analyzed snapshot image.

## 5. Exiting
- To exit the GUI overlay or raw feed frame cleanly, press `q`.
- Alternatively, force quit via terminal `CTRL+C`.
