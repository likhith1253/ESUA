# Deployment Guide

Although ESUA is inherently built for local webcam-based analysis, it can be deployed to cloud or web architectures if you want to expose its functionality over the internet (e.g., parsing user-uploaded images or handling remote video streaming).

## Deployment Platforms

### Option A: Render (Web Service)
If you want to wrap ESUA in a Flask/FastAPI interface and host it online:
1. Add a `Dockerfile` to leverage the official `python:3.9-slim` image. Be sure to `apt-get install -y libgl1-mesa-glx libglib2.0-0` to satisfy OpenCV backend dependencies.
2. Ensure you initialize the `yolov8n.pt` weight fetch dynamically prior to server start, or bake it directly into the image to avoid cold-start degradation.
3. Deploy as a Web Service. Note: **Free-tier Render** will spin down after 15 minutes of inactivity; the YOLO model requires several seconds to boot from a cold start.

### Option B: Edge Devices (Raspberry Pi & Jetson Nano)
ESUA is heavily optimized for localized IoT edge implementation.
1. The standard `yolov8n.pt` will achieve 3-5 FPS on a Raspberry Pi 4 CPU. 
2. **NVIDIA Jetson:** If deploying to a Jetson Nano, explicitly compile `ultralytics` pointing to `torchvision` matched precisely with your local JetPack OS install. This engages TensorRT acceleration for robust real-time performance.
3. ESUA handles frame skipping inherently via the `SKIP_FRAMES` logic inside `camera_runner.py` to prevent queue bottlenecking on lower-end SoCs.

## Environmental Configuration
If pushing to a cloud-based Platform as a Service (PaaS):

- **Headless Mode:** Do **not** attempt to trigger `cv2.imshow()`, `cv2.waitKey()`, or `cv2.destroyAllWindows()` in a hosted environment. Comment out GUI overlay commands as PaaS frameworks do not ship with dedicated display interfaces and will crash.
- **Port Handling:** If deploying as a FastAPI wrapper, verify `$PORT` environment variables are mapped accurately against the host configuration.

## Vercel Constraints
Vercel is primarily built for Serverless Edge caching. Given that YOLO model weights natively exceed standard serverless bundle size ceilings and PyTorch operations severely outstrip Vercel’s execution timeout window limit, **Vercel is not recommended** for backend ESUA hosting. A persistent container solution like Render, AWS EC2, or Google Cloud Run is required.
