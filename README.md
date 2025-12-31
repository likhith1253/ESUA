# Phase 6: Camera Integration

This folder integrates real-time camera logic into the **Explainable Scene Understanding Assistant (ESUA)**. It connects the previous logic layers (Spatial & Risk) to a live video feed.

## Files

### 1. `camera_runner.py` (Real-Time Mode)
A live monitoring script that processes the video feed in real-time. It runs ESUA analysis on the fly to detect and warn about risks immediately.
- **How to Run:** `python ESUA/phase6_camera_integration/camera_runner.py`
- **Behavior:** Wraps objects in boxes and overlays risk warnings (e.g., "Spill Risk") on the screen.
- **Controls:** Press `q` to quit.

### 2. `snapshot_analyzer.py` (High Accuracy Mode)
A more precise tool that captures a "burst" of 5 frames to confirm objects before analyzing them.
- **How to Run:** `python ESUA/phase6_camera_integration/snapshot_analyzer.py`
- **Behavior:** Reduces false detections ("ghost objects") by ensuring an object appears in multiple frames before accepting it.
- **Controls:** Press `c` to capture a snapshot for analysis, `q` to quit.

## Dependencies
- **Webcam**: Ensure a camera is connected.
- **Libraries**: `opencv-python`, `ultralytics`.
