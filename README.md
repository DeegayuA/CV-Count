# CV-Count 🚶‍♂️📈
**CV-Count** is an interactive, high-performance People Counter using YOLO and ByteTrack. Designed for flexibility, it supports multiple camera angles (including top-down), hardware acceleration, and optimized low-latency processing.

## ✨ Key Features
- **Smart Setup**: Automatically detects NVIDIA RTX/GTX GPUs and installs the correct CUDA-accelerated environment.
- **Premium UI & UX**: Modern Slate & Emerald theme with smooth hover animations and a live help bar.
- **Pixel-Perfect Setup**: Draw lines and zones directly on a resized preview; the app handles all background math to maintain full resolution accuracy.
- **Interactive Multi-Model Support**: Switch between YOLOv11 and YOLOv26 on the fly.
- **High Performance**: Optimized for NVIDIA RTX (CUDA) and Apple Silicon (Metal/MPS) with automatic detection.
- **Graceful Control**: Close the app cleanly using hotkeys or the window [X] button.

---

## 🚀 Getting Started

### 1. Requirements
- **Hardware Acceleration**: 
    - **NVIDIA GPU**: Automatically configures the ~2.6GB CUDA package.
    - **Apple Silicon**: Automatically configures Metal (MPS) support.

### 2. Smart Installation
Run the provided installer:
- **`02. Setup (Run first).bat`**: This will scan your hardware, ask for permission for large downloads, and configure everything automatically.

### 3. Usage
- **`03. Start CV Count App (Run after 02).bat`**: Launches the main interface.

---

## 🖱️ Controls & HUD
- **[N]**: Start drawing a counting **Line**. (Click 2 points)
- **[Z]**: Start drawing a counting **Zone** (Polygon). (Click multiple points)
- **[Backspace]**: Undo the last point.
- **[C]**: Clear all lines and zones.
- **[SPACE]**: Start counting!
- **[Q/Esc]**: Exit application.
- **[X]**: Corner button to close the window.
- **Tooltips**: Hover over any setting in the menu to see a description in the status bar at the bottom.

---

## 📂 Project Structure
- `assets/`: Put your videos here! Output results are also saved here.
- `models/`: YOLO models are automatically downloaded and stored here.
- `setup_env.py`: The intelligent hardware scanner and installer.
- `counter.py`: The main application logic.
