# CV-Count 🚶‍♂️📈
**CV-Count** is an interactive, high-performance People Counter using YOLO and ByteTrack. Designed for flexibility, it supports multiple camera angles (including top-down), hardware acceleration, and optimized low-latency processing.

---

## 🚀 Getting Started

### 1. Requirements
- **Python 3.10+** (Already configured for `C:\Python314\python.exe` in the Batch files).
- **GPU (Optional but Recommended)**: NVIDIA CUDA (RTX/GTX) or Apple Metal.
  - If you have an NVIDIA GPU (like an RTX 3050) but the app says "CUDA missing", run this command:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
    ```

### 2. Quick Install
Run the provided setup script:
`📄 02. Setup (Run first).bat`

### 3. Start the App
Run the start script:
`📄 03. Start CV Count App (Run after 02).bat`

---

## 🔧 Workflow

### Step 1: Settings Screen
Configure your detection parameters:
- **Source Selection**: Pick from the `assets/` folder, select a **Live Camera (0, 1, 2)**, or browse any local file.
- **Model Selection**: 11 or 26 YOLO models ranging from Nano (fast) to XLarge (expert).
- **Optimization**: Select **Inference Resolution** (320px for speed, 1280px for distant people).
- **Angle Support**: Choose between **Center Point** (best for top-down) or **Base/Feet** (best for ground-level).

### Step 2: Setup Phase
On the first frame of the video/camera:
- **Draw Lines**: Press `[N]` then click two points.
- **Draw Zones**: Press `[Z]` then click points for a polygon. (People are only counted if they cross within/through this zone).
- **Customize**: Press `[F]` to flip the counting direction of the last line.
- **Undo**: Press `[Backspace]` to remove the last point or line.
- **Start**: Press `[SPACE]` to begin counting.

### Step 3: Processing
- Watch live counting with a transparent HUD.
- **Automatic Output**: If "Save Video" is enabled, results are stored in the `assets/` folder as `(file)_counted.mp4`.

---

## ⌨️ Shortcuts & Controls

| Phase | Action | Hotkey |
| :--- | :--- | :--- |
| **Settings** | UI Selections | `🖱️ Click` |
| **Settings** | Exit Application | `Q` or `Esc` |
| **Setup** | Draw Line Mode | `N` |
| **Setup** | Draw Zone Mode | `Z` |
| **Setup** | Flip Line Direction | `F` |
| **Setup** | Undo Last Point/Line | `Backspace` |
| **Setup** | Clear All Drawings | `C` |
| **Setup** | Start Processing | `Space` |
| **Active** | Pause / Resume | `Space` |
| **Active** | Stop / Quit | `Q` or `Esc` |
| **Active** | Reset Counts | `R` |

---

## 💎 Premium Features
- **Hardware Acceleration (Auto-Detect)**: Automatically maps weights to CUDA, MPS (Apple), or CPU.
- **Dual-Res Pipeline**: Performs "heavy lifting" (Detection) at lower resolution while keeping the visual output at full native resolution (1080p/4K).
- **Persistence Tuning**: Uses an optimized ByteTrack configuration with a 120-frame buffer to ensure people aren't double-counted during occlusion.
- **Multi-Angle Logic**: Specialized Support for front, back, side, and steep top-down camera views.

---

## 📁 File Structure
- `counter.py`: Main interactive application.
- `assets/`: Default folder for input videos and saved results.
- `models/`: Automatically managed folder for YOLO weights.
- `bytetrack.yaml`: High-persistence tracker configuration.
- `02. Setup...bat`: Integrated installer (pip + dependencies).
