# Local Setup Guide

Follow this guide to get ESUA running on your local machine. Because this project is computationally lightweight, it does not mandate a dedicated GPU, though CUDA or MPS execution is supported if appropriately configured through PyTorch.

## Prerequisites

- **Python:** `3.8`, `3.9`, or `3.10` recommended
- **Git:** For cloning the repository
- **Operational Webcam:** Required for real-time analysis modules

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ESUA.git
cd ESUA
```

### 2. Prepare Virtual Environment (Recommended)
Set up an isolated python environment to avoid package conflicts.
```bash
# Windows
python -m venv esua-env
esua-env\Scripts\activate

# macOS / Linux
python3 -m venv esua-env
source esua-env/bin/activate
```

### 3. Install Dependencies
The base requirements are minimalistic:
```bash
pip install -r requirements.txt
```
This automatically handles fetching the standard Ultralytics package (which includes PyTorch), OpenCV-python for camera streaming, and requests.

### 4. Verify Model Download
Ensure the `yolov8n.pt` weights file is located in the root codebase directory (`d:\ESUA\`). If it is not present, the `ultralytics` package should automatically attempt to pull the weights upon initial execution.

### 5. Verify Hardware Permissions
Depending on your OS (especially Windows 11 and macOS), Python will need explicit permission to access the local camera hardware. Ensure these permissions are granted in your system's privacy settings.

### Custom Configuration Preparation
To experiment with custom sample data without a live camera, you can place test images inside the `ESUA/phase1_object_detection/` and `ESUA/phase2_spatial_understanding/` directories as `sample.jpg`.
