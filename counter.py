# pyre-ignore-all-errors
"""
CV-Count — Interactive Line-Crossing People Counter
====================================================
UI IMPROVEMENTS:
  • Settings screen with Video Picker (click to browse)
  • Fixed overlap (title and subtitle)
  • START button position fixed
  • Polished visuals + Drag-and-Drop capability (via Explorer)

FLOW:
  1. Pick Video, Model, and Mode on the on-screen UI
  2. Setup on first frame: Draw lines (N) or Zones (Z)
  3. SPACE to start counting
"""

import sys, os, collections, time, tkinter as tk
from tkinter import filedialog
import cv2  # type: ignore
import numpy as np  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    sys.exit("[ERROR] Run: RUN_CV_COUNT.bat")

try:
    import lap  # type: ignore
except ImportError:
    sys.exit("[ERROR] Run: RUN_CV_COUNT.bat")


# ─────────────────────── CONFIG ──────────────────────────────────────────────

ASSETS_DIR     = "assets"
MODELS_DIR     = "models"
DEFAULT_VIDEO  = os.path.join(ASSETS_DIR, "01.30fps.mp4")
CLASSES        = [0]
CONF           = 0.35
IOU            = 0.45
TRACKER        = "bytetrack.yaml"
DRAW_TRACKS    = True
TRAIL_LEN      = 30
SKIP_FRAMES    = 0
LINE_THICKNESS = 3
HUD_ALPHA      = 0.70
MAX_WIN_W      = 1280
MAX_WIN_H      = 850

# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────── helpers ─────────────────────────────────────────────

_PALETTE = [
    (0,200,255),(0,255,128),(255,80,80),(255,50,220),
    (80,255,255),(180,80,255),(80,200,80),(255,160,80),
]

def _side(pt,a,b):
    return (b[0]-a[0])*(pt[1]-a[1])-(b[1]-a[1])*(pt[0]-a[0])

def _get_poi(box, point_type="center"):
    """Get Point of Interest (PoI) for tracking."""
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if point_type == "base":
        return int((x1+x2)/2), int(y2)
    return int((x1+x2)/2), int((y1+y2)/2)

def _puttext(img,txt,pos,scale=0.48,color=(210,210,220),thick=1,bold=False):
    cv2.putText(img,txt,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,
                int(thick) + (1 if bold else 0),cv2.LINE_AA)

def _in_rect(x,y,rx,ry,rw,rh):
    return rx<=x<=rx+rw and ry<=y<=ry+rh

def _inside_zone(pt, zone_poly):
    if not zone_poly or len(zone_poly) < 3: return True
    poly = np.array(zone_poly, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(pt[0]),float(pt[1])), False) >= 0

def _detect_device():
    """Detect available GPU/NPU acceleration for YOLO inference."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            # NVIDIA CUDA
            return "NVIDIA CUDA (RTX/GTX)", "0"
        
        # Diagnostics
        has_gpu_hw = False
        try:
            import subprocess
            res = subprocess.run(["nvidia-smi"], capture_output=True)
            if res.returncode == 0: has_gpu_hw = True
        except: pass

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "Apple Metal (MPS)", "mps"
        
        if torch.version.cuda is None and has_gpu_hw:
            return "CPU (GPU Found but Torch is CPU-only)", "cpu"
        elif has_gpu_hw:
            # Maybe driver is missing or mismatch
            return "CPU (NVIDIA Found but Driver/CUDA Error)", "cpu"
        
        return "CPU (No Acceleration found)", "cpu"
    except ImportError:
        return "CPU (PyTorch not found)", "cpu"
    except: pass
    return "CPU (No Acceleration found)", "cpu"


# ─────────────────────── settings screen ─────────────────────────────────────

_MODELS = [
    ("YOLO26 Nano",   "yolo26n.pt"),
    ("YOLO26 Small",  "yolo26s.pt"),
    ("YOLO26 Medium", "yolo26m.pt"),
    ("YOLO26 Large",  "yolo26l.pt"),
    ("YOLO26 XLarge", "yolo26x.pt"),
    ("YOLO11 Nano",   "yolo11n.pt"),
    ("YOLO11 Medium", "yolo11m.pt"),
    ("YOLOv8 Nano",   "yolov8n.pt"),
    ("YOLOv8 Medium", "yolov8m.pt"),
]
_MODES = [
    ("Both directions  --  adds +1 either way",       "both_add"),
    ("One-way only     --  adds +1 for IN side only", "one_way"),
    ("Net occupancy    --  adds IN +1 / OUT -1",      "net"),
]
_LATEST = {"yolo26n.pt","yolo26s.pt","yolo26m.pt","yolo26l.pt","yolo26x.pt"}

# 🎨 THEME: SLATE & EMERALD
_BG=(15,12,10); _CARD=(28,24,20); _ACC=(129,185,16); _INF=(212,182,6); _GRN=(0,255,0)
_SEL=(60,45,20); _HOV=(45,35,25); _WHITE=(252,250,248); _MUT=(148,115,100); _RED=(0,0,255)
_YEL=(0,255,255); _BLU=(255,150,0); _TXT=(215,215,225)

def _mdl_cached(fn):
    return os.path.isfile(os.path.join(MODELS_DIR,fn)) or (os.path.isabs(fn) and os.path.isfile(fn))

def pick_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    init_dir = os.path.abspath(ASSETS_DIR) if os.path.isdir(ASSETS_DIR) else None
    fpath = filedialog.askopenfilename(title="Select Video File", 
                                       initialdir=init_dir,
                                       filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
    root.destroy()
    return fpath

def settings_screen():
    SW,SH = 900, 1020
    WIN   = "CV-Count -- Settings"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, SW, SH)

    hw_label, device_id = _detect_device()

    # Pre-scan assets (Auto-create if missing)
    if not os.path.exists(ASSETS_DIR):
        try: 
            os.makedirs(ASSETS_DIR, exist_ok=True)
            with open(os.path.join(ASSETS_DIR, "put your videos here.txt"), "w") as f:
                f.write("Place your video files (.mp4, .avi, etc.) in this folder to have them show up in the app dropdown.")
        except: pass

    afiles = []
    try:
        if os.path.isdir(ASSETS_DIR):
            all_f = os.listdir(ASSETS_DIR)
            afiles = sorted([str(f) for f in all_f if f.lower().endswith(('.mp4','.avi','.mkv','.mov'))])
            if len(afiles) > 6: afiles = afiles[:6]  # type: ignore
    except: pass

    st = {
        "video": DEFAULT_VIDEO,
        "model": 2,
        "mode": 0,
        "imgsz": 640,
        "conf": 0.35,
        "point": "center",
        "augment": False,
        "save": True,
        "start": False
    }
    mouse = {"x":0,"y":0,"dn":False}

    def on_mouse(ev,x,y,fl,_):
        mouse["x"]=x; mouse["y"]=y
        if ev==cv2.EVENT_LBUTTONDOWN: mouse["dn"]=True
    cv2.setMouseCallback(WIN,on_mouse)

    def render(hm,hmo,help_msg=""):
        img=np.full((SH,SW,3),_BG,dtype=np.uint8)
        
        # Header
        cv2.rectangle(img,(0,0),(SW,65),_CARD,-1)
        cv2.line(img,(0,65),(SW,65),_ACC,1)
        _puttext(img,"CV-COUNT",(25,45),0.95,_ACC,bold=True)
        _puttext(img,"Interactive Line Counter  |  v1.3",(175,42),0.42,_MUT)
        
        # Acceleration Banner
        tx = SW - 350
        cv2.rectangle(img,(tx-10,12),(SW-25,52),(0,40,45) if device_id!="cpu" else (10,10,25),-1)
        cv2.rectangle(img,(tx-10,12),(SW-25,52),_GRN if device_id!="cpu" else (50,50,60),1)
        _puttext(img,"ACCEL:",(tx,35),0.35,_MUT)
        _puttext(img,hw_label,(tx+55,35),0.40,_GRN if device_id!="cpu" else (80,80,220),bold=device_id!="cpu")

        def _sec(y,lbl):
            cv2.rectangle(img,(25,y),(SW-25,y+24),_CARD,-1)
            _puttext(img,lbl,(34,y+18),0.44,_ACC,bold=True)

        # ───────────────── SOURCE SELECTION ──────────────
        _sec(85, "INPUT SOURCE (ASSETS / CAMERA / EXTERNAL)")
        cv2.rectangle(img,(35,120),(SW-35,215),_CARD,-1); cv2.rectangle(img,(35,120),(SW-35,215),_MUT,1)
        _puttext(img,"FOLDER: assets/ (Auto-detects files)",(45,138),0.35,_MUT)
        CW_A = (SW-70)//2
        afiles_sub = list(afiles)
        for i,f in enumerate(afiles_sub[:8]): # type: ignore
            col, row = i%2, i//2
            rx, ry = 40 + col*CW_A, 153 + row*18
            sel=st["video"]==os.path.join(ASSETS_DIR,f)
            if sel: cv2.rectangle(img,(rx-5,ry-13),(rx+CW_A-10,ry+2),_SEL,-1)
            _puttext(img,f[:45],(rx+5,ry),0.38,_WHITE if sel else _TXT)
            
        # Camera & Browse underneath to use horizontal space well
        _puttext(img,"LIVE CAMERAS:",(35, 240),0.35,_MUT)
        for i in range(3):
            rx=125+i*85
            sel=st["video"]==i
            cv2.rectangle(img,(rx,226),(rx+75,256),_SEL if sel else _CARD,-1)
            cv2.rectangle(img,(rx,226),(rx+75,256),_ACC if sel else _MUT,1)
            _puttext(img,f"CAM {i}",(rx+15,246),0.40,_WHITE if sel else _TXT)
            
        bx,by=390,226
        hv_b = _in_rect(mouse["x"],mouse["y"],bx,by,SW-bx-35,30)
        cv2.rectangle(img,(bx,by),(SW-35,by+30),_HOV if hv_b else _CARD,-1)
        cv2.rectangle(img,(bx,by),(SW-35,by+30),_ACC if hv_b else _MUT,1)
        _puttext(img,"BROWSE EXPLORER...",(bx+25,by+21),0.42,_WHITE if hv_b else _TXT)

        # Current Source Label
        if isinstance(st['video'], int):
            slbl = f"SELECTED: Camera {st['video']}"
        else:
            v_name = os.path.basename(str(st['video']))
            slbl = f"SELECTED: {v_name}"
        
        slbl_s = str(slbl)
        if len(slbl_s) > 95: slbl_s = slbl_s[:95]  # type: ignore
        _puttext(img,slbl_s,(35,245),0.40,_GRN)

        # ───────────────── MODEL SECTION ─────────────────
        _sec(285, "DETECTION MODEL")
        RH=35; CW=(SW-70)//2
        for i,(lbl,fn) in enumerate(_MODELS):
            col, row = i%2, i//2
            rx, ry = 35 + col*CW, 320 + row*RH
            selected=i==st["model"]; hovered=i==hm and not selected
            bg=_SEL if selected else (_HOV if hovered else _BG)
            cv2.rectangle(img,(rx,ry),(rx+CW-5,ry+RH-2),bg,-1)
            bx,bc=rx+20,ry+RH//2
            if selected: 
                cv2.circle(img,(bx,bc),8,_ACC,-1); cv2.circle(img,(bx,bc),4,_BG,-1)
            else: 
                cv2.circle(img,(bx,bc),8,_MUT,1)
            _puttext(img,lbl,(rx+40,ry+RH//2+6),0.48, _WHITE if selected else _TXT)
            
            # Tags
            tx=rx+40+len(str(lbl))*10+5
            if _mdl_cached(fn):
                cv2.rectangle(img,(tx,ry+8),(tx+55,ry+RH-10),(0,65,30),-1)
                _puttext(img,"CACHED",(tx+4,ry+RH-13),0.32,_GRN)
            elif fn in _LATEST:
                cv2.rectangle(img,(tx,ry+8),(tx+55,ry+RH-10),(0,45,75),-1)
                _puttext(img,"LATEST",(tx+4,ry+RH-13),0.32,_ACC)

        # ───────────────── DETECTION SETTINGS ────────────
        _sec(485, "OPTIMIZATION & QUALITY")  # SHIFTED DOWN
        # Resolution
        _puttext(img,"Detection Res:",(35,535),0.42,_MUT)
        res_opts = [320, 640, 1280]
        for i,rv in enumerate(res_opts):
            rx=160+i*110
            sel=st["imgsz"]==rv
            cv2.rectangle(img,(rx,518),(rx+100,548),_SEL if sel else _CARD,-1)
            cv2.rectangle(img,(rx,518),(rx+100,548),_ACC if sel else _MUT,1)
            _puttext(img,f"{rv}px",(rx+22,538),0.42,_WHITE if sel else _TXT)

        # Confidence
        _puttext(img,"Confidence:",(520,535),0.42,_MUT)
        conf_opts = [0.20, 0.35, 0.50]
        for i,cv in enumerate(conf_opts):
            rx=620+i*80
            sel=abs(st["conf"]-cv)<0.01
            cv2.rectangle(img,(rx,518),(rx+70,548),_SEL if sel else _CARD,-1)
            cv2.rectangle(img,(rx,518),(rx+70,548),_ACC if sel else _MUT,1)
            _puttext(img,f"{cv:.2f}",(rx+15,538),0.42,_WHITE if sel else _TXT)

        # Counting Point & Augment
        _sec(585, "ANGLE & POSE SUPPORT")  # SHIFTED DOWN
        # Point
        _puttext(img,"Counting Point:",(35,635),0.42,_MUT)
        for i,pv in enumerate(["center","base"]):
            rx=165+i*125
            sel=st["point"]==pv
            lbl="CENTER" if pv=="center" else "BASE (FEET)"
            cv2.rectangle(img,(rx,618),(rx+115,648),_SEL if sel else _CARD,-1)
            cv2.rectangle(img,(rx,618),(rx+115,648),_ACC if sel else _MUT,1)
            _puttext(img,lbl,(rx+12,637),0.38,_WHITE if sel else _TXT)
        _puttext(img,"(Use CENTER for top-down shots)",(165,662),0.32,_MUT)
        
        # Augment
        _puttext(img,"Advanced Angle Support:",(500,635),0.42,_MUT)
        ax=715
        cv2.rectangle(img,(ax,618),(SW-35,648),_SEL if st["augment"] else _CARD,-1)
        cv2.rectangle(img,(ax,618),(SW-35,648),_ACC if st["augment"] else _MUT,1)
        _puttext(img,"ENHANCED (TTA)" if st["augment"] else "STANDARD (OFF)",(ax+15,637),0.38,_GRN if st["augment"] else _MUT)

        # ───────────────── MODE SECTION ──────────────────
        _sec(705, "COUNTING RULES")  # SHIFTED DOWN
        for i,(lbl,_) in enumerate(_MODES):
            ry=740+i*RH
            selected = i == st.get("mode")
            hovered = i == hmo and not selected
            bg=_SEL if selected else (_HOV if hovered else _BG)
            cv2.rectangle(img,(35,ry),(SW-35,ry+RH-2),bg,-1)
            bx,bc=55,ry+RH//2
            if selected: 
                cv2.circle(img,(bx,bc),8,_ACC,-1); cv2.circle(img,(bx,bc),4,_BG,-1)
            else: 
                cv2.circle(img,(bx,bc),8,_MUT,1)
            _puttext(img,lbl,(75,ry+RH//2+6),0.50,_WHITE if selected else _TXT)

        # ───────────────── START BUTTON ──────────────────
        STBY=SH-92
        hs=_in_rect(mouse["x"],mouse["y"],SW//2-130,STBY,260,65)
        # Pulse-style glow
        import time
        glow = int(10 * np.sin(time.time()*4)) + 10
        cv2.rectangle(img,(SW//2-130,STBY),(SW//2+130,STBY+65),_ACC,-1)
        if hs: cv2.rectangle(img,(SW//2-135,STBY-2),(SW//2+135,STBY+67),_WHITE,2)
        _puttext(img,"LAUNCH PROJECT",(SW//2-92,STBY+42),0.72,_WHITE,bold=True)

        # Help Tooltip Area
        cv2.rectangle(img,(0,SH-30),(SW,SH),_CARD,-1)
        _puttext(img,help_msg if help_msg else "Q/Esc to exit  |  Interactive People Counting v1.6",(25,SH-10),0.36,_MUT if not help_msg else _INF)
        return img

    while True:
        mx,my=mouse["x"],mouse["y"]
        hm, hmo = -1, -1
        help_msg = ""
        CW=(SW-70)//2
        
        # Hover Detection & Tooltips
        if _in_rect(mx,my,25,85,SW-50,180): help_msg = "Select a video from assets, a live camera, or browse your PC."
        
        for i in range(len(_MODELS)):
            col, row = i%2, i//2
            if _in_rect(mx,my,35+col*CW,320+row*35,CW-5,33): 
                hm=i; help_msg = f"Use {_MODELS[i][0]} for optimized counting."
                break
        
        if _in_rect(mx,my,25,485,SW-50,80): help_msg = "Resolution: 320 (Fastest) -> 1280 (High Detail). Confidence: Filter noise."
        if _in_rect(mx,my,25,585,SW-50,100): help_msg = "Center: Ideal for Top-Down cameras. Base: Ideal for ground-level views."
        
        for i in range(len(_MODES)):
            if _in_rect(mx,my,35,740+i*35,SW-70,33): 
                hmo=i; help_msg = "Choose how movements are tracked: Sum, Subtract (Net), or One-Way."
                break
        
        hv_b = _in_rect(mx,my,475,195,SW-475-35,30)

        if mouse["dn"]:  # type: ignore
            mouse["dn"]=False  # type: ignore
            # Assets clicks
            afiles_sub2 = list(afiles)
            for i,f in enumerate(afiles_sub2[:8]):  # type: ignore
                col, row = i%2, i//2
                CW_A = (SW-70)//2
                rx, ry = 40 + col*CW_A, 153 + row*18
                if _in_rect(mx,my,rx-5,ry-13,CW_A-10,18): st["video"]=os.path.join(ASSETS_DIR,f)
            # Camera clicks
            for i in range(3):
                if _in_rect(mx,my,125+i*85,226,75,30): st["video"]=i
            # Browse click
            if hv_b:
                new_v = pick_file()
                if new_v: st["video"]=new_v
            
            elif hm>=0: st["model"]=hm
            elif hmo>=0: st["mode"]=hmo
            elif _in_rect(mx,my,SW//2-130,SH-92,260,65): st["start"]=True  # type: ignore
            else:
                # Resolution & Confidence clicks
                for i,rv in enumerate([320,640,1280]):
                    if _in_rect(mx,my,160+i*110,518,100,32): st["imgsz"]=rv
                for i,cv in enumerate([0.20,0.35,0.50]):
                    if _in_rect(mx,my,620+i*80,518,70,32): st["conf"]=cv
                # Point clicks
                for i,pv in enumerate(["center","base"]):
                    if _in_rect(mx,my,165+i*125,618,115,30): st["point"]=pv
                # Augment click
                if _in_rect(mx,my,715,618,150,30): st["augment"]=not st["augment"]  # type: ignore

        cv2.imshow(WIN,render(hm,hmo,help_msg))
        key=cv2.waitKey(20)&0xFF
            
        if key in (ord('q'),27): cv2.destroyAllWindows(); sys.exit(0)
        if st["start"]: break  # type: ignore

    cv2.destroyWindow(WIN)
    return st["video"], _MODELS[int(st["model"])][1], _MODES[int(st["mode"])][1], int(st["imgsz"]), float(st["conf"]), device_id, st["point"], st["augment"]


# ─────────────────────── common drawing ───────────────────────────────────────

def _draw_zone(img, zone_poly, in_progress=False, cursor=None):
    if not zone_poly: return
    pts = np.array(zone_poly, dtype=np.int32)
    # Emerald Green styling
    if not in_progress and len(pts) >= 3:
        ov = img.copy()
        cv2.fillPoly(ov, [pts], _GRN)
        cv2.addWeighted(ov, 0.15, img, 0.85, 0, img)
        cv2.polylines(img, [pts], True, _GRN, 2, cv2.LINE_AA)
    else:
        for i in range(len(pts)-1):
            cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), _GRN, 2, cv2.LINE_AA)
        if cursor and len(pts)>0:
            cv2.line(img, tuple(pts[-1]), cursor, _GRN, 1, cv2.LINE_AA)
    for p in zone_poly:
        cv2.circle(img, p, 5, _WHITE, -1)

def _draw_lines(img, counting_lines, hover_idx=-1):
    for i, ln in enumerate(counting_lines):
        a, b = ln["p1"], ln["p2"]
        # Yellow line with highlight if hovered
        color = _YEL
        thick = LINE_THICKNESS + (2 if i == hover_idx else 0)
        cv2.line(img, a, b, color, thick, cv2.LINE_AA)
        cv2.circle(img, a, 5, color, -1)
        cv2.circle(img, b, 5, color, -1)
        
        # Bi-Color Directional Arrows
        mid = ((a[0]+b[0])//2, (a[1]+b[1])//2)
        dx, dy = b[0]-a[0], b[1]-a[1]
        mag = np.sqrt(dx**2 + dy**2)
        if mag > 0:
            # Perpendicular vector for IN/OUT markers
            nx, ny = -dy/mag, dx/mag
            off = 25
            
            # IN direction (P1 to P2 right-side)
            in_p = (int(mid[0] + nx*off), int(mid[1] + ny*off))
            cv2.arrowedLine(img, mid, in_p, _GRN, 2, tipLength=0.4)
            _puttext(img, "IN", (in_p[0]+5, in_p[1]), 0.35, _GRN, bold=True)
            
            # OUT direction
            out_p = (int(mid[0] - nx*off), int(mid[1] - ny*off))
            cv2.arrowedLine(img, mid, out_p, _RED, 2, tipLength=0.4)
            _puttext(img, "OUT", (out_p[0]+5, out_p[1]), 0.35, _RED, bold=True)
            
        # Line Number Label
        _puttext(img, f"#{i+1}", (a[0]+10, a[1]-10), 0.45, _YEL, bold=True)

def _draw_hud(frame, counting_lines, count_mode, zone_active, fps_val=0):
    rows = 1 + len(counting_lines) + (1 if zone_active else 0) + 2
    ph, pw = rows*32+20, 260
    ov = frame.copy()
    cv2.rectangle(ov,(10,10),(10+pw,10+ph),(15,15,18),-1)
    
    if fps_val > 0:
        cv2.rectangle(ov, (10+pw-75, 10), (10+pw, 35), (30,120,50), -1)
        
    cv2.addWeighted(ov,HUD_ALPHA,frame,1-HUD_ALPHA,0,frame)
    _puttext(frame, "CV-COUNT DASHBOARD", (20, 32), 0.45, _ACC, bold=True)
    
    if fps_val > 0:
        _puttext(frame, f"{fps_val:02} FPS", (10+pw-68, 27), 0.45, _WHITE, bold=True)
    
    yc=int(65); grand=0
    if zone_active:
        _puttext(frame, "ZONE MASK: ACTIVE", (20, yc), 0.4, _GRN); yc+=32
        
    for i,ln in enumerate(counting_lines):
        v = (ln["in"]+ln["out"]) if count_mode=="both_add" else (ln["in"] if count_mode=="one_way" else ln["in"]-ln["out"])
        grand+=v
        cv2.circle(frame,(28,yc-6),7,ln["color"],-1)
        _puttext(frame, f"Line {i+1}:", (45, yc), 0.48, (230,230,230))
        t = f"{v:+d}" if count_mode=="net" else str(v)
        _puttext(frame, t, (pw-35, yc), 0.55, ln["color"], bold=True)
        yc+=32
    
    cv2.line(frame,(20,yc-15),(pw,yc-15),(70,70,75),1)
    _puttext(frame, "GRAND TOTAL:", (20, yc+10), 0.52, (240,240,240), bold=True)
    gt = f"{grand:+d}" if count_mode=="net" else str(grand)
    cv2.putText(frame, gt, (pw-35, yc+12), cv2.FONT_HERSHEY_DUPLEX, 0.7, _GRN, 2)


# ─────────────────────── main ─────────────────────────────────────────────────

def main():
    video_path, model_name, count_mode, imgsz, conf, dev_id, poi_type, use_aug = settings_screen()

    # Model load
    os.makedirs(MODELS_DIR, exist_ok=True)
    mp = model_name if os.path.isabs(model_name) else os.path.join(MODELS_DIR, model_name)
    _orig = os.getcwd(); os.chdir(MODELS_DIR)
    model = YOLO(os.path.basename(mp))
    if dev_id != "cpu": model.to(dev_id)
    os.chdir(_orig)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): sys.exit(f"[ERROR] Cannot open {video_path}")
    fw, fh = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    totf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_live = totf <= 0
    
    scale = min(MAX_WIN_W/fw, MAX_WIN_H/fh, 1.0)
    dw, dh = int(fw*scale), int(fh*scale)

    writer = None
    os.makedirs(ASSETS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    temp_outp = os.path.join(ASSETS_DIR, f"{base}_temp.mp4")
    final_outp = os.path.join(ASSETS_DIR, f"{base}_counted.mp4")
    writer = cv2.VideoWriter(temp_outp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw,fh))

    # state
    counting_lines, zone_poly = [], []
    zone_drawing, line_drawing, draw_start = False, False, None
    prev_sides = {}
    trails = collections.defaultdict(lambda: collections.deque(maxlen=TRAIL_LEN))
    mouse_pos = [0,0]

    WIN = "CV-Count"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, dw, dh)

    # Setup Phase on a high-fidelity original frame
    ret, frame1 = cap.read()
    if not ret: sys.exit("[ERROR] Cannot read first frame of video.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def on_m(ev,x,y,fl,_):
        nonlocal zone_drawing, line_drawing, draw_start
        mouse_pos[0], mouse_pos[1] = x, y
        
        if ev == cv2.EVENT_LBUTTONDOWN:
            if zone_drawing:
                zone_poly.append((x,y))
            elif line_drawing:
                if draw_start is None:
                    draw_start=(x,y)
                else:
                    c = _PALETTE[len(counting_lines) % len(_PALETTE)]
                    counting_lines.append({"p1":draw_start, "p2":(x,y), "color":c, "in":0, "out":0})
                    draw_start = None # Stay in line mode
        elif ev == cv2.EVENT_LBUTTONDBLCLK:
            if zone_drawing and len(zone_poly) >= 3:
                zone_drawing = False
    cv2.setMouseCallback(WIN, on_m)
    
    while True:
        f = frame1.copy()
        
        # Smart Hover: Identify closest line for Smart Flip (F)
        mx, my = mouse_pos
        hover_idx = -1
        if counting_lines:
            best_d = 50 # Max distance to be considered "hovered"
            for i, ln in enumerate(counting_lines):
                ax, ay = ln["p1"]
                bx, by = ln["p2"]
                midx, midy = (ax+bx)/2, (ay+by)/2
                d = np.sqrt((mx-midx)**2 + (my-midy)**2)
                if d < best_d:
                    best_d, hover_idx = d, i

        _draw_zone(f, zone_poly, zone_drawing, tuple(mouse_pos))
        _draw_lines(f, counting_lines, hover_idx)
        
        # Visual crosshair
        cv2.drawMarker(f, (mx,my), _WHITE, cv2.MARKER_CROSS, 20, 1)

        if line_drawing and draw_start:
            cv2.circle(f, draw_start, 8, _YEL, -1)  # type: ignore
            _puttext(f, "CLICK FOR END POINT", (int(draw_start[0]+15), int(draw_start[1]-15)), 0.65, _YEL, bold=True)  # type: ignore
        
        # Banner UI
        bh = fh // 20
        banner_bg = _BLU
        if zone_drawing: banner_bg = _GRN
        elif line_drawing: banner_bg = _YEL
        
        cv2.rectangle(f, (0, fh-bh), (fw, fh), banner_bg, -1)
        
        mod = "ZONE (DBL-CLICK TO FINISH)" if zone_drawing else ("LINE (CONTINUOUS)" if line_drawing else "READY")
        banner = f" [{mod}] -- [N] Line | [Z] Zone | [F] Flip Line under Mouse | [C] CLEAR | [SPACE] START "
        _puttext(f, banner, (30, fh - (bh//2) + 12), 0.75 * (fh/1080), _BG if line_drawing else _WHITE, bold=True)
        
        cv2.imshow(WIN, f)
        
        k = cv2.waitKey(1)&0xFF
        if k in (ord('q'), 27): return
        elif k==ord('n'): line_drawing=True; zone_drawing=False; draw_start=None
        elif k==ord('z'): 
            if zone_drawing: zone_drawing=False # Toggle
            else: zone_drawing=True; line_drawing=False; draw_start=None
        elif k==ord('f'):
            if hover_idx >= 0:
                ln = counting_lines[hover_idx]
                ln["p1"], ln["p2"] = ln["p2"], ln["p1"]
        elif k==8: # Backspace
            if zone_drawing and zone_poly: zone_poly.pop()
            elif counting_lines: counting_lines.pop()
        elif k==ord('c'): counting_lines.clear(); zone_poly.clear(); draw_start=None
        elif k==ord(' '):
            if counting_lines or zone_poly: break

    # Process Phase
    paused, frame_idx, last_f = False, 0, None
    zone_active = len(zone_poly)>=3

    fps_start = float(time.time())
    fc = 0
    cur_fps = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break
            last_f = frame.copy()
            fc = fc + 1  # type: ignore
            if time.time() - fps_start > 0.5:  # type: ignore
                cur_fps = int(fc / (time.time() - fps_start))  # type: ignore
                fps_start = time.time()
                fc = 0
        else: frame = last_f.copy() if last_f is not None else np.zeros((10,10,3), dtype=np.uint8)  # type: ignore

        if not paused:
            # OPTIMIZATION: Detection on resized resolution (imgsz), overlapping results back to full res.
            res = model.track(frame, imgsz=imgsz, conf=conf, iou=IOU, persist=True, tracker=TRACKER, verbose=False, classes=CLASSES, device=dev_id, augment=use_aug)
            if zone_active: _draw_zone(frame, zone_poly)
            
            if res and res[0].boxes is not None:
                for b in res[0].boxes:
                    xy = b.xyxy[0].cpu().numpy().astype(int)
                    tid = int(b.id[0]) if b.id is not None else -1
                    cx, cy = _get_poi(xy, poi_type)
                    inside = _inside_zone((cx,cy), zone_poly)
                    if not inside: continue
                    
                    cv2.rectangle(frame, (xy[0],xy[1]), (xy[2],xy[3]), _GRN if inside else (50,50,220), 1)
                    if DRAW_TRACKS and tid!=-1:
                        trails[tid].append((cx,cy))  # type: ignore
                        pts = list(trails[tid])  # type: ignore
                        for pi in range(1,len(pts)):
                            cv2.line(frame, pts[pi-1], pts[pi], (0,200,80) if inside else (40,40,200), 1)
                    
                    if tid!=-1:
                        if tid not in prev_sides: prev_sides[tid]={}  # type: ignore
                        for li,ln in enumerate(counting_lines):
                            cur = _side((cx,cy), ln["p1"], ln["p2"])
                            prev = prev_sides[tid].get(li)  # type: ignore
                            if prev is not None and prev!=0 and cur!=0 and (prev>0)!=(cur>0):
                                if inside:
                                    if count_mode=="both_add": ln["in" if prev>0 else "out"]+=1
                                    elif count_mode=="one_way" and prev>0: ln["in"]+=1
                                    elif count_mode=="net": ln["in" if prev>0 else "out"]+=1
                            if cur!=0: prev_sides[tid][li] = cur  # type: ignore
                            
        _draw_lines(frame, counting_lines)
        _draw_hud(frame, counting_lines, count_mode, zone_active, cur_fps)
        if paused: _puttext(frame, "PAUSED", (fw//2-100, fh//2), 2.0, _ACC, 4)
        
        if writer: writer.write(frame)  # type: ignore
        cv2.imshow(WIN, frame)
        k = cv2.waitKey(1)&0xFF
        if k in (ord('q'),27): break
        elif k==ord(' '): paused = not paused
        elif k==ord('r'):
            for ln in counting_lines: ln["in"]=ln["out"]=0
            prev_sides.clear()

    cap.release()
    if writer: writer.release()  # type: ignore
    cv2.destroyAllWindows()

    # ─────────────────────── POST-PROCESS REPORT ────────────────────────────────

    WIN_REP = "Final Report"
    cv2.namedWindow(WIN_REP, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_REP, 700, 500)
    
    saved = False
    done = False
    
    def on_m_rep(ev,x,y,fl,_):
        nonlocal saved, done
        if ev == cv2.EVENT_LBUTTONDOWN:
            if _in_rect(x,y, 100, 380, 200, 50): # SAVE
                saved = True; done = True
            elif _in_rect(x,y, 400, 380, 200, 50): # DISCARD
                saved = False; done = True
    cv2.setMouseCallback(WIN_REP, on_m_rep)
    
    while not done:
        img = np.full((500, 700, 3), _BG, dtype=np.uint8)
        _puttext(img, "FINAL COUNT REPORT", (180, 80), 1.0, _ACC, bold=True)
        
        grand = 0
        yc = 150
        for i, ln in enumerate(counting_lines):
            v = (ln["in"]+ln["out"]) if count_mode=="both_add" else (ln["in"] if count_mode=="one_way" else ln["in"]-ln["out"])
            grand += v
            _puttext(img, f"Line {i+1}:", (200, yc), 0.7, (230,230,230))
            _puttext(img, str(v), (450, yc), 0.8, ln.get("color", _YEL), bold=True)
            yc += 45
        
        cv2.line(img, (150, yc), (550, yc), (70,70,75), 1)
        yc += 40
        _puttext(img, "GRAND TOTAL:", (200, yc), 0.8, _WHITE, bold=True)
        _puttext(img, str(grand), (450, yc), 1.0, _GRN, bold=True)
        
        # SAVE BTN
        cv2.rectangle(img, (100, 380), (300, 430), _GRN, -1)
        _puttext(img, "SAVE VIDEO", (135, 412), 0.6, _WHITE, bold=True)
        # DISCARD BTN
        cv2.rectangle(img, (400, 380), (600, 430), (50,50,60), -1)
        _puttext(img, "DISCARD", (445, 412), 0.6, _WHITE, bold=True)
        
        cv2.imshow(WIN_REP, img)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 32): 
            saved = True; done = True
        elif k in (27, ord('q')): 
            saved = False; done = True
            
        if cv2.getWindowProperty(WIN_REP, cv2.WND_PROP_AUTOSIZE) == -1: break
            
    cv2.destroyAllWindows()
    
    if saved:
        if os.path.exists(final_outp): os.remove(final_outp)
        os.rename(temp_outp, final_outp)
    else:
        if os.path.exists(temp_outp): os.remove(temp_outp)

if __name__=="__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
