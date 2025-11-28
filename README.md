# Auto-Faceblur - Computer Vision

Automatically detect and blur faces in images and video using DeepFace and OpenCV.

Overview
- Detect faces in images or video frames using DeepFace's face detectors (or other detectors as available).
- Blur detected faces (Gaussian or pixelation) to anonymize people.
- Simple CLI / script examples for processing single images and videos.

Features
- Image and video processing
- Confidence threshold for face detections
- Choice of blur method and blur strength
- Works with CPU and (when installed) GPU-accelerated TensorFlow

Requirements
- Python 3.8+
- See requirements.txt for Python packages used by this project.

Quick start

1. Clone the repository
   git clone https://github.com/One-Of-Those-Organization/auto-faceblur.git
   cd auto-faceblur

2. Create a virtual environment (recommended)
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows PowerShell

3. Install dependencies
   pip install -r requirements.txt

Note on tf-keras
- The requirements file lists `tf-keras` as requested. If `tf-keras` is unavailable on your environment or you prefer the main TensorFlow package, install `tensorflow` (and optionally `keras`) instead. For GPU support, install the appropriate TensorFlow GPU package for your system.
