# Phase 6: Camera Integration

This folder contains the real-time camera integration for the **Explainable Scene Understanding Assistant (ESUA)**.

## Overview
The `camera_runner.py` script captures live video from your webcam, runs the full ESUA pipeline (Detection -> Spatial -> Risk -> Explanation) on every 5th frame, and overlays the results directly on the video feed.

## How to Run
1.  Ensure you have a webcam connected.
2.  Run the script from the root directory:
    ```bash
    python ESUA/phase6_camera_integration/camera_runner.py
    ```
3.  Press **'q'** to quit the application.

## Key Features
-   **Real-Time Detection**: Uses YOLOv8n to find objects.
-   **Live Risk Assessment**: Automatically flags risks like "Cup near Laptop".
-   **On-Screen Explanations**: Displays simplified warnings directly on the video.
-   **Performance Optimized**: Skips frames to ensure the video stays smooth even on CPU.

## Troubleshooting
-   **"Could not open webcam"**: Check if another app (like Zoom or Teams) is using the camera.
-   **Low FPS**: The script is optimized for CPU, but older machines might still struggle. Ensure strictly no other heavy apps are running.
