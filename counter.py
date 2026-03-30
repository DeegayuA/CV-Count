"""
CV-Count — Interactive Line-Crossing People Counter
=====================================================
Usage:
    python counter.py

On startup you will be prompted to choose:
  • Detection model
  • Counting mode
  • Whether to display the live window
  • Whether to save output video

Controls (once running):
    N        → Enter "draw line" mode (click 2 points to place a line)
    C        → Clear all lines and reset counts
    R        → Reset counts only (keep lines)
    Space    → Pause / Resume
    Q / Esc  → Quit and save output video
"""

# ─────────────────────── STATIC CONFIG ──────────────────────────────────────
# These are defaults / advanced options not asked at startup.

VIDEO_PATH     = "01.50fps.mp4.mp4"   # path to video, or 0 for webcam
MODELS_DIR     = "models"              # folder where .pt weights are stored
CLASSES        = [0]                  # [0] = person only; [] = all classes
CONF           = 0.35                 # detection confidence threshold (0–1)
IOU            = 0.45                 # NMS IoU threshold
TRACKER        = "bytetrack.yaml"     # bytetrack.yaml or botsort.yaml
DRAW_TRACKS    = True                 # draw centroid trails on frame
TRAIL_LEN      = 30                   # number of frames to keep in trail
SKIP_FRAMES    = 0                    # process every N+1 frames (0 = every frame)
LINE_THICKNESS = 3                    # drawn counting line thickness (px)
HUD_ALPHA      = 0.60                 # HUD panel transparency (0=invisible,1=opaque)

# ─────────────────────── COUNTING MODES ──────────────────────────────────────
# Selected at runtime; kept here for documentation.
#   "both_add"  → crossing in EITHER direction adds +1
#   "one_way"   → only crossing in the "IN" direction adds +1
#   "net"       → IN crossing adds +1, OUT crossing subtracts -1 (net occupancy)

# ─────────────────────────────────────────────────────────────────────────────

import sys, os, collections
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] ultralytics not installed.\n  Run:  C:\\Python314\\python.exe -m pip install -r requirements.txt")

# 'lap' is required by ByteTrack for linear assignment (matching.py).
try:
    import lap  # noqa: F401
except ImportError:
    sys.exit(
        "[ERROR] 'lap' is not installed (required by ByteTrack).\n"
        "  Run:  C:\\Python314\\python.exe -m pip install lap"
    )



# ─────────────────────── palette & helpers ───────────────────────────────────

_PALETTE = [
    (0, 200, 255), (0, 255, 128), (255, 80, 80),  (255, 50, 220),
    (80, 255, 255),(180, 80, 255),(80, 200, 80),   (255, 160, 80),
]

def _side(pt, a, b):
    return (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])

def _centroid(box):
    x1,y1,x2,y2 = box
    return int((x1+x2)/2), int((y1+y2)/2)


# ─────────────────────── startup prompts ─────────────────────────────────────

def _model_icon(filename):
    """✓ if the .pt already exists in MODELS_DIR (or as an absolute path), ○ otherwise."""
    if filename in ("__custom__", ""):
        return "  "
    # absolute / relative path given by user
    if os.path.isabs(filename) and os.path.isfile(filename):
        return "✓ "
    # check models folder
    if os.path.isfile(os.path.join(MODELS_DIR, filename)):
        return "✓ "
    return "○ "

def _pick(prompt, options, default_idx=0, show_icons=False):
    """Display a numbered menu and return the user's choice value."""
    print(prompt)
    if show_icons:
        print("  (✓ = already downloaded  |  ○ = will download on first use)")
    for i, (label, val) in enumerate(options):
        icon   = _model_icon(val) if show_icons else ""
        marker = " [default]" if i == default_idx else ""
        print(f"  {i+1:2}. {icon}{label}{marker}")
    while True:
        raw = input(f"  Enter choice [1-{len(options)}] (Enter = {default_idx+1}): ").strip()
        if raw == "":
            return options[default_idx][1]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw)-1][1]
        print("  Invalid choice, try again.")

def _ask_yn(question, default="y"):
    hint = "[Y/n]" if default=="y" else "[y/N]"
    raw  = input(f"  {question} {hint}: ").strip().lower()
    if raw == "":
        return default == "y"
    return raw in ("y","yes")

def startup_config():
    """Interactive prompts; returns (model_name, count_mode, show_window, output_path)."""
    print("\n" + "="*58)
    print("  CV-Count — Interactive Line-Crossing Counter")
    print("="*58)

    # ── model ─────────────────────────────────────────────────────────────────
    model_name = _pick("\n[1/3] Select detection model:", [
        ("YOLO26 Nano   ★ LATEST — NMS-free, fastest CPU (Jan 2026)", "yolo26n.pt"),
        ("YOLO26 Small  ★ LATEST — fast + accurate",                  "yolo26s.pt"),
        ("YOLO26 Medium ★ LATEST — balanced",                         "yolo26m.pt"),
        ("YOLO26 Large  ★ LATEST — most accurate",                    "yolo26l.pt"),
        ("YOLO11 Nano   (legacy, ~39 MB)",                            "yolo11n.pt"),
        ("YOLO11 Small  (legacy, ~77 MB)",                            "yolo11s.pt"),
        ("YOLO11 Medium (legacy, ~170 MB)",                           "yolo11m.pt"),
        ("YOLOv8 Nano   (legacy, ~6 MB)",                             "yolov8n.pt"),
        ("YOLOv8 Medium (legacy, ~52 MB)",                            "yolov8m.pt"),
        ("Custom — type path below",                                   "__custom__"),
    ], default_idx=0, show_icons=True)
    if model_name == "__custom__":
        model_name = input("  Enter model path/name: ").strip()

    # ── counting mode ─────────────────────────────────────────────────────────
    count_mode = _pick("\n[2/3] Select counting mode:", [
        ("Both directions add  (+1 either way)",        "both_add"),
        ("One-way only         (+1 for IN direction)",  "one_way"),
        ("Net occupancy        (IN=+1 / OUT=−1)",       "net"),
    ], default_idx=0)

    # ── display & save ────────────────────────────────────────────────────────
    print("\n[3/3] Output options:")
    show_window  = _ask_yn("Show live video window?", default="y")
    save_video   = _ask_yn("Save processed video?",   default="y")
    output_path  = None
    if save_video:
        base    = os.path.splitext(os.path.basename(VIDEO_PATH if not isinstance(VIDEO_PATH, int) else "webcam"))[0]
        default_out = f"{base}_counted.mp4"
        raw = input(f"  Output filename [Enter = {default_out}]: ").strip()
        output_path = raw if raw else default_out

    return model_name, count_mode, show_window, output_path


# ─────────────────────── HUD renderer ────────────────────────────────────────

def _draw_hud(frame, counting_lines, count_mode):
    h, w = frame.shape[:2]
    pad, row_h, margin = 10, 28, 10

    # header + per-line rows + separator + total
    rows      = 1 + len(counting_lines) + 1 + 1
    panel_h   = rows * row_h + 2 * pad
    panel_w   = 240
    x0, y0   = margin, margin

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+panel_w, y0+panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, HUD_ALPHA, frame, 1-HUD_ALPHA, 0, frame)

    mode_label = {"both_add":"BOTH DIRS +1","one_way":"ONE-WAY +1","net":"NET ±1"}[count_mode]
    cv2.putText(frame, f"CV-COUNT  [{mode_label}]", (x0+pad, y0+pad+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160,160,160), 1, cv2.LINE_AA)

    y_cur = y0 + pad + row_h
    grand = 0
    for i, ln in enumerate(counting_lines):
        color = ln["color"]
        if count_mode == "both_add":
            val  = ln["in"] + ln["out"]
            txt  = str(val)
        elif count_mode == "one_way":
            val  = ln["in"]
            txt  = str(val)
        else:  # net
            val  = ln["in"] - ln["out"]
            txt  = f"{val:+d}" if val != 0 else "0"
        grand += val

        cv2.circle(frame, (x0+pad+8, y_cur+4), 7, color, -1)
        cv2.putText(frame, f"Line {i+1}:", (x0+pad+22, y_cur+9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200,200,200), 1, cv2.LINE_AA)
        # direction sub-counts for info
        sub = f"({ln['in']}↓ {ln['out']}↑)" if count_mode != "one_way" else f"({ln['in']}↓)"
        cv2.putText(frame, sub, (x0+pad+22, y_cur+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120,120,120), 1, cv2.LINE_AA)
        val_x = x0 + panel_w - pad - max(len(txt)*11, 20)
        cv2.putText(frame, txt, (val_x, y_cur+9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y_cur += row_h

    cv2.line(frame, (x0+pad, y_cur), (x0+panel_w-pad, y_cur), (70,70,70), 1)
    y_cur += row_h - 8
    cv2.putText(frame, "TOTAL:", (x0+pad, y_cur+9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220,220,220), 1, cv2.LINE_AA)
    g_txt = f"{grand:+d}" if count_mode=="net" and grand != 0 else str(grand)
    gx = x0 + panel_w - pad - max(len(g_txt)*14, 20)
    cv2.putText(frame, g_txt, (gx, y_cur+9),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0,255,180), 2, cv2.LINE_AA)


def _draw_lines(frame, counting_lines):
    for i, ln in enumerate(counting_lines):
        a, b = ln["p1"], ln["p2"]
        cv2.line(frame, a, b, ln["color"], LINE_THICKNESS, cv2.LINE_AA)
        mid = ((a[0]+b[0])//2, (a[1]+b[1])//2)
        # draw direction arrow (small tick on the "IN" side)
        dx, dy = b[0]-a[0], b[1]-a[1]
        length  = max((dx**2+dy**2)**0.5, 1)
        nx, ny  = int(-dy/length*18), int(dx/length*18)
        cv2.arrowedLine(frame, (mid[0], mid[1]),
                        (mid[0]+nx, mid[1]+ny),
                        (255,255,255), 1, tipLength=0.4)
        cv2.putText(frame, f"L{i+1}", (mid[0]+8, mid[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ln["color"], 2, cv2.LINE_AA)


# ─────────────────────── main ─────────────────────────────────────────────────

def main():
    model_name, count_mode, show_window, output_path = startup_config()

    # ── load model from / into MODELS_DIR ────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.isabs(model_name):
        model_path = model_name          # absolute custom path — use as-is
    else:
        model_path = os.path.join(MODELS_DIR, model_name)
    print(f"\n[INFO] Loading model: {model_path} …")
    # YOLO downloads the weights to the parent directory of the given path
    # when the file doesn't exist yet, so we temporarily chdir.
    _orig_dir = os.getcwd()
    os.chdir(MODELS_DIR)
    model = YOLO(os.path.basename(model_path))
    os.chdir(_orig_dir)
    print("[INFO] Model ready.\n")

    print("Controls:")
    print("  N=new line  |  C=clear all  |  R=reset counts  |  Space=pause  |  Q/Esc=quit\n")

    # ── open video ────────────────────────────────────────────────────────────
    src = VIDEO_PATH
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open: {src}")

    fw          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src     = cap.get(cv2.CAP_PROP_FPS) or 30
    tot_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {fw}×{fh} @ {fps_src:.1f} fps  ({tot_frames} frames)")

    # ── output writer ─────────────────────────────────────────────────────────
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_src, (fw, fh))
        print(f"[INFO] Saving output → {output_path}")

    # ── state ─────────────────────────────────────────────────────────────────
    counting_lines = []
    prev_sides     = {}
    trails         = collections.defaultdict(lambda: collections.deque(maxlen=TRAIL_LEN))
    draw_mode      = False
    draw_start     = None
    paused         = False
    frame_idx      = 0
    last_frame     = None

    win_name = "CV-Count  [N=line | C=clear | R=reset | Space=pause | Q=quit]"
    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(fw, 1280), min(fh, 720))

    # ── mouse callback ────────────────────────────────────────────────────────
    def on_mouse(event, x, y, flags, param):
        nonlocal draw_mode, draw_start
        if not draw_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if draw_start is None:
                draw_start = (x, y)
                print(f"[LINE] P1={draw_start} — click P2")
            else:
                color = _PALETTE[len(counting_lines) % len(_PALETTE)]
                counting_lines.append({"p1": draw_start, "p2": (x,y),
                                       "color": color, "in": 0, "out": 0})
                print(f"[LINE] Line {len(counting_lines)} → {draw_start} to ({x},{y})")
                draw_start = None
                draw_mode  = False

    if show_window:
        cv2.setMouseCallback(win_name, on_mouse)

    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        if not paused:
            for _ in range(max(1, SKIP_FRAMES+1)):
                ret, frame = cap.read()
                if not ret:
                    frame = None
                    break
                frame_idx += 1
            if frame is None:
                print("[INFO] End of video.")
                break
            last_frame = frame.copy()
        else:
            frame = last_frame.copy() if last_frame is not None else \
                    np.zeros((fh, fw, 3), dtype=np.uint8)

        # ── inference ─────────────────────────────────────────────────────────
        if not paused:
            kw = dict(conf=CONF, iou=IOU, persist=True,
                      tracker=TRACKER, verbose=False)
            if CLASSES:
                kw["classes"] = CLASSES
            results = model.track(frame, **kw)
        else:
            results = None

        # ── process detections ────────────────────────────────────────────────
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy  = box.xyxy[0].cpu().numpy().astype(int)
                tid   = int(box.id[0]) if box.id is not None else -1
                cx,cy = _centroid(xyxy)

                # bounding box
                cv2.rectangle(frame, (xyxy[0],xyxy[1]), (xyxy[2],xyxy[3]),
                               (50,200,50), 1)
                if tid != -1:
                    cv2.putText(frame, f"#{tid}", (xyxy[0]+4, xyxy[1]-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50,255,100), 1, cv2.LINE_AA)

                # trail
                if DRAW_TRACKS and tid != -1:
                    trails[tid].append((cx,cy))
                    pts = list(trails[tid])
                    for pi in range(1, len(pts)):
                        a2 = pi/len(pts)
                        cv2.line(frame, pts[pi-1], pts[pi],
                                 (int(50*a2),int(255*a2),int(100*a2)), 1, cv2.LINE_AA)

                # crossing
                if tid == -1:
                    continue
                if tid not in prev_sides:
                    prev_sides[tid] = {}

                for li, ln in enumerate(counting_lines):
                    cur  = _side((cx,cy), ln["p1"], ln["p2"])
                    prev = prev_sides[tid].get(li)

                    if prev is not None and prev != 0 and cur != 0 and (prev>0) != (cur>0):
                        # crossed — determine direction
                        going_in = (prev > 0)   # left-of-line → right = "in"
                        if count_mode == "both_add":
                            if going_in:
                                counting_lines[li]["in"]  += 1
                            else:
                                counting_lines[li]["out"] += 1
                        elif count_mode == "one_way":
                            if going_in:
                                counting_lines[li]["in"]  += 1
                            # out direction ignored
                        else:  # net
                            if going_in:
                                counting_lines[li]["in"]  += 1
                            else:
                                counting_lines[li]["out"] += 1

                    if cur != 0:
                        prev_sides[tid][li] = cur

        # ── draw ──────────────────────────────────────────────────────────────
        _draw_lines(frame, counting_lines)

        # in-progress draw feedback
        if draw_mode and draw_start:
            cv2.circle(frame, draw_start, 6, (0,255,255), -1)
            cv2.putText(frame, "Click P2", (draw_start[0]+10, draw_start[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
        elif draw_mode:
            cv2.putText(frame, "Click Point 1 (line start)",
                        (20, fh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,255,255), 2, cv2.LINE_AA)

        if counting_lines:
            _draw_hud(frame, counting_lines, count_mode)

        # progress bar
        if tot_frames > 0 and not paused:
            prog = int(fw * frame_idx / tot_frames)
            cv2.rectangle(frame, (0, fh-4), (prog, fh), (0,200,255), -1)

        if paused:
            cv2.putText(frame, "PAUSED", (fw//2-70, fh//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (0,200,255), 3, cv2.LINE_AA)

        # ── output ────────────────────────────────────────────────────────────
        if writer:
            writer.write(frame)
        if show_window:
            cv2.imshow(win_name, frame)

        # ── keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF if show_window else 0xFF

        if key in (ord('q'), 27):
            print("[INFO] Quit.")
            break
        elif key == ord('n'):
            draw_mode  = True
            draw_start = None
            print("[LINE] Draw mode ON — click 2 points.")
        elif key == ord('c'):
            counting_lines.clear(); prev_sides.clear(); trails.clear()
            draw_mode = False; draw_start = None
            print("[INFO] Cleared all lines and counts.")
        elif key == ord('r'):
            for ln in counting_lines:
                ln["in"] = ln["out"] = 0
            prev_sides.clear()
            print("[INFO] Counts reset.")
        elif key == ord(' '):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}.")

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"[INFO] Saved: {output_path}")

    # final report
    print("\n" + "="*44)
    print("  FINAL COUNTS")
    print(f"  Mode: {count_mode}")
    print("="*44)
    grand = 0
    for i, ln in enumerate(counting_lines):
        if count_mode == "both_add":
            val = ln["in"] + ln["out"]
            info = f"(↓{ln['in']} ↑{ln['out']})"
        elif count_mode == "one_way":
            val  = ln["in"]
            info = ""
        else:
            val  = ln["in"] - ln["out"]
            info = f"(in={ln['in']} out={ln['out']})"
        grand += val
        print(f"  Line {i+1}: {val:>5}  {info}")
    print(f"  {'─'*34}")
    print(f"  TOTAL  : {grand}")
    print("="*44 + "\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
