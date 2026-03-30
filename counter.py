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

# ─────────────────────── CONFIG ──────────────────────────────────────────────

DEFAULT_VIDEO  = "01.50fps.mp4.mp4"
MODELS_DIR     = "models"
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

import sys, os, collections, tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] Run: RUN_CV_COUNT.bat")

try:
    import lap  # noqa: F401
except ImportError:
    sys.exit("[ERROR] Run: RUN_CV_COUNT.bat")


# ─────────────────────── helpers ─────────────────────────────────────────────

_PALETTE = [
    (0,200,255),(0,255,128),(255,80,80),(255,50,220),
    (80,255,255),(180,80,255),(80,200,80),(255,160,80),
]

def _side(pt,a,b):
    return (b[0]-a[0])*(pt[1]-a[1])-(b[1]-a[1])*(pt[0]-a[0])

def _centroid(box):
    x1,y1,x2,y2=box
    return int((x1+x2)/2),int((y1+y2)/2)

def _puttext(img,txt,pos,scale=0.48,color=(210,210,220),thick=1,bold=False):
    cv2.putText(img,txt,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,
                thick+(1 if bold else 0),cv2.LINE_AA)

def _in_rect(x,y,rx,ry,rw,rh):
    return rx<=x<=rx+rw and ry<=y<=ry+rh

def _inside_zone(pt, zone_poly):
    if not zone_poly or len(zone_poly) < 3: return True
    poly = np.array(zone_poly, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(pt[0]),float(pt[1])), False) >= 0


# ─────────────────────── settings screen ─────────────────────────────────────

_MODELS = [
    ("YOLO26 Nano",   "yolo26n.pt"),
    ("YOLO26 Small",  "yolo26s.pt"),
    ("YOLO26 Medium", "yolo26m.pt"),
    ("YOLO26 Large",  "yolo26l.pt"),
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
_LATEST = {"yolo26n.pt","yolo26s.pt","yolo26m.pt","yolo26l.pt"}

_BG=(18,18,22); _CARD=(28,28,36); _ACC=(0,190,255); _GRN=(0,220,110)
_SEL=(0,65,100); _HOV=(40,40,55); _TXT=(215,215,225); _MUT=(115,115,130)

def _mdl_cached(fn):
    return os.path.isfile(os.path.join(MODELS_DIR,fn)) or (os.path.isabs(fn) and os.path.isfile(fn))

def pick_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    fpath = filedialog.askopenfilename(title="Select Video File", 
                                       filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
    root.destroy()
    return fpath

def settings_screen():
    SW,SH = 900, 780
    WIN   = "CV-Count -- Settings"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, SW, SH)

    st = {
        "video": DEFAULT_VIDEO,
        "model": 2,
        "mode": 0,
        "save": True,
        "start": False
    }
    mouse = {"x":0,"y":0,"dn":False}

    def on_mouse(ev,x,y,fl,_):
        mouse["x"]=x; mouse["y"]=y
        if ev==cv2.EVENT_LBUTTONDOWN: mouse["dn"]=True
    cv2.setMouseCallback(WIN,on_mouse)

    def render(hm,hmo,hv):
        img=np.full((SH,SW,3),_BG,dtype=np.uint8)
        
        # Header
        cv2.rectangle(img,(0,0),(SW,65),_CARD,-1)
        cv2.line(img,(0,65),(SW,65),_ACC,1)
        _puttext(img,"CV-COUNT",(25,45),0.95,_ACC,bold=True)
        _puttext(img,"Interactive Line Counter  |  v1.2",(175,42),0.42,_MUT)

        def _sec(y,lbl):
            cv2.rectangle(img,(25,y),(SW-25,y+24),_CARD,-1)
            _puttext(img,lbl,(34,y+18),0.44,_ACC,bold=True)

        # ───────────────── VIDEO SECTION ─────────────────
        _sec(85, "VIDEO SOURCE")
        v_h = _in_rect(mouse["x"], mouse["y"], 35, 120, SW-70, 42)
        cv2.rectangle(img, (35,120), (SW-35,162), _HOV if v_h else _BG, -1)
        cv2.rectangle(img, (35,120), (SW-35,162), _MUT if not v_h else _ACC, 1)
        v_lbl = os.path.basename(st["video"]) if st["video"] else "CLICK TO SELECT VIDEO"
        _puttext(img, v_lbl, (55, 147), 0.52, _WHITE if v_h else _TXT)
        _puttext(img, "(Click to change file)", (SW-165, 147), 0.38, _MUT)

        # ───────────────── MODEL SECTION ─────────────────
        _sec(185, "DETECTION MODEL")
        RH=35
        for i,(lbl,fn) in enumerate(_MODELS):
            ry=220+i*RH
            selected=i==st["model"]; hovered=i==hm and not selected
            bg=_SEL if selected else (_HOV if hovered else _BG)
            cv2.rectangle(img,(35,ry),(SW-35,ry+RH-2),bg,-1)
            bx,bc=55,ry+RH//2
            if selected: 
                cv2.circle(img,(bx,bc),8,_ACC,-1); cv2.circle(img,(bx,bc),4,_BG,-1)
            else: 
                cv2.circle(img,(bx,bc),8,_MUT,1)
            _puttext(img,lbl,(75,ry+RH//2+6),0.50,(255,255,255) if selected else _TXT)
            
            # Tags
            tx=75+len(lbl)*10+10
            if _mdl_cached(fn):
                cv2.rectangle(img,(tx,ry+8),(tx+65,ry+RH-10),(0,65,30),-1)
                _puttext(img,"CACHED",(tx+6,ry+RH-13),0.35,_GRN)
            elif fn in _LATEST:
                cv2.rectangle(img,(tx,ry+8),(tx+65,ry+RH-10),(0,45,75),-1)
                _puttext(img,"LATEST",(tx+6,ry+RH-13),0.35,_ACC)

        # ───────────────── MODE SECTION ──────────────────
        _sec(515, "COUNTING RULES")
        for i,(lbl,_) in enumerate(_MODES):
            ry=550+i*RH
            selected=i==st["mode"]; hovered=i==hmo and not selected
            bg=_SEL if selected else (_HOV if hovered else _BG)
            cv2.rectangle(img,(35,ry),(SW-35,ry+RH-2),bg,-1)
            bx,bc=55,ry+RH//2
            if selected: 
                cv2.circle(img,(bx,bc),8,_ACC,-1); cv2.circle(img,(bx,bc),4,_BG,-1)
            else: 
                cv2.circle(img,(bx,bc),8,_MUT,1)
            _puttext(img,lbl,(75,ry+RH//2+6),0.50,(255,255,255) if selected else _TXT)

        # ───────────────── SAVE SECTION ──────────────────
        _sec(670, "SAVE OPTIONS")
        yb=705
        yc=_SEL if st["save"] else _CARD
        nc=(60,35,70) if not st["save"] else _CARD
        cv2.rectangle(img,(45,yb),(155,yb+32),yc,-1)
        cv2.rectangle(img,(45,yb),(155,yb+32),_ACC if st["save"] else _MUT,1)
        _puttext(img,"SAVE VIDEO",(65,yb+23),0.42,_GRN if st["save"] else _MUT)
        
        cv2.rectangle(img,(170,yb),(280,yb+32),nc,-1)
        cv2.rectangle(img,(170,yb),(280,yb+32),_MUT,1)
        _puttext(img,"NO SAVE",(195,yb+23),0.42,(180,80,180) if not st["save"] else _MUT)

        # ───────────────── START BUTTON ──────────────────
        STBY=SH-85
        hs=_in_rect(mouse["x"],mouse["y"],SW//2-130,STBY,260,58)
        cv2.rectangle(img,(SW//2-130,STBY),(SW//2+130,STBY+58),(0,180,95) if hs else (0,150,80),-1)
        cv2.rectangle(img,(SW//2-130,STBY),(SW//2+130,STBY+58),_GRN,1)
        _puttext(img,"LAUNCH PROJECT",(SW//2-88,STBY+37),0.68,_WHITE,bold=True)

        _puttext(img,"Q/Esc to exit  |  Auto-downloads models as needed",(25,SH-15),0.36,_MUT)
        return img

    while True:
        mx,my=mouse["x"],mouse["y"]
        hm=-1
        for i in range(len(_MODELS)):
            if _in_rect(mx,my,35,220+i*35,SW-70,33): hm=i; break
        hmo=-1
        for i in range(len(_MODES)):
            if _in_rect(mx,my,35,550+i*35,SW-70,33): hmo=i; break
        hv = _in_rect(mx,my,35,120,SW-70,42)

        if mouse["dn"]:
            mouse["dn"]=False
            if hv:
                new_v = pick_file()
                if new_v: st["video"]=new_v
            elif hm>=0: st["model"]=hm
            elif hmo>=0: st["mode"]=hmo
            elif _in_rect(mx,my,45,705,110,32):  st["save"]=True
            elif _in_rect(mx,my,170,705,110,32): st["save"]=False
            elif _in_rect(mx,my,SW//2-130,SH-85,260,58): st["start"]=True

        cv2.imshow(WIN,render(hm,hmo,hv))
        key=cv2.waitKey(15)&0xFF
        if key in (ord('q'),27): cv2.destroyAllWindows(); sys.exit(0)
        if st["start"]: break

    cv2.destroyWindow(WIN)
    return st["video"], _MODELS[st["model"]][1], _MODES[st["mode"]][1], st["save"]


# ─────────────────────── common drawing ───────────────────────────────────────

def _draw_zone(img, zone_poly, in_progress=False, cursor=None):
    if not zone_poly: return
    pts = np.array(zone_poly, dtype=np.int32)
    if not in_progress and len(pts) >= 3:
        ov = img.copy()
        cv2.fillPoly(ov, [pts], (0,80,20))
        cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)
        cv2.polylines(img, [pts], True, (0,230,100), 2, cv2.LINE_AA)
    else:
        for i in range(len(pts)-1):
            cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), (0,200,80), 2, cv2.LINE_AA)
        if cursor and len(pts)>0:
            cv2.line(img, tuple(pts[-1]), cursor, (0,200,80), 1, cv2.LINE_AA)
    for p in zone_poly:
        cv2.circle(img, p, 5, (0,250,110), -1)

def _draw_lines(img, counting_lines):
    for i, ln in enumerate(counting_lines):
        a,b = ln["p1"],ln["p2"]
        cv2.line(img,a,b,ln["color"],LINE_THICKNESS,cv2.LINE_AA)
        mid = ((a[0]+b[0])//2,(a[1]+b[1])//2)
        dx,dy = b[0]-a[0],b[1]-a[1]
        L = max((dx**2+dy**2)**0.5,1)
        nx,ny = int(-dy/L*20),int(dx/L*20)
        cv2.arrowedLine(img,mid,(mid[0]+nx,mid[1]+ny),(255,255,255),2,tipLength=0.3)
        _puttext(img,f"L{i+1}",(mid[0]+10,mid[1]-10),0.6,ln["color"],bold=True)

def _draw_hud(frame, counting_lines, count_mode, zone_active):
    rows = 1 + len(counting_lines) + (1 if zone_active else 0) + 2
    ph, pw = rows*32+20, 260
    ov = frame.copy()
    cv2.rectangle(ov,(10,10),(10+pw,10+ph),(15,15,18),-1)
    cv2.addWeighted(ov,HUD_ALPHA,frame,1-HUD_ALPHA,0,frame)
    _puttext(frame, "CV-COUNT DASHBOARD", (20, 32), 0.45, _ACC, bold=True)
    
    yc=65; grand=0
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
    video_path, model_name, count_mode, do_save = settings_screen()

    # Model load
    os.makedirs(MODELS_DIR, exist_ok=True)
    mp = model_name if os.path.isabs(model_name) else os.path.join(MODELS_DIR, model_name)
    _orig = os.getcwd(); os.chdir(MODELS_DIR)
    model = YOLO(os.path.basename(mp))
    os.chdir(_orig)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): sys.exit(f"[ERROR] Cannot open {video_path}")
    fw, fh = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    totf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    scale = min(MAX_WIN_W/fw, MAX_WIN_H/fh, 1.0)
    dw, dh = int(fw*scale), int(fh*scale)

    writer = None
    if do_save:
        base = os.path.splitext(os.path.basename(video_path))[0]
        outp = f"{base}_counted.mp4"
        writer = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw,fh))

    # state
    counting_lines, zone_poly = [], []
    zone_drawing, line_drawing, draw_start = False, False, None
    prev_sides = {}
    trails = collections.defaultdict(lambda: collections.deque(maxlen=TRAIL_LEN))
    mouse_pos = [0,0]

    WIN = "CV-Count"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, dw, dh)

    def on_m(ev,x,y,fl,_):
        nonlocal zone_drawing, line_drawing, draw_start
        mouse_pos[0], mouse_pos[1] = x, y
        if ev==cv2.EVENT_LBUTTONDOWN:
            if zone_drawing: zone_poly.append((x,y))
            elif line_drawing:
                if draw_start is None: draw_start=(x,y)
                else:
                    c=_PALETTE[len(counting_lines)%len(_PALETTE)]
                    counting_lines.append({"p1":draw_start,"p2":(x,y),"color":c,"in":0,"out":0})
                    draw_start=None; line_drawing=False
    cv2.setMouseCallback(WIN, on_m)

    # Setup Phase
    ret, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        f = frame1.copy()
        _draw_zone(f, zone_poly, zone_drawing, tuple(mouse_pos))
        _draw_lines(f, counting_lines)
        if line_drawing and draw_start:
            cv2.circle(f,draw_start,6,(0,255,255),-1)
            _puttext(f,"Click P2",(draw_start[0]+12,draw_start[1]-12),0.5,(0,255,255))
        
        banner = " SETUP: [N] Draw Line  |  [Z] Draw Zone  |  [SPACE] Start "
        _puttext(f, banner, (20, fh-25), 0.65, _ACC, bold=True)
        
        cv2.imshow(WIN, f)
        k = cv2.waitKey(20)&0xFF
        if k in (ord('q'),27): return
        elif k==ord('n'): line_drawing=True; zone_drawing=False; draw_start=None
        elif k==ord('z'): 
            if zone_drawing and len(zone_poly)>=3: zone_drawing=False
            else: zone_drawing=True; line_drawing=False
        elif k==ord('x'): zone_poly.clear(); zone_drawing=False
        elif k==ord('c'): counting_lines.clear(); zone_poly.clear()
        elif k==ord(' '):
            if counting_lines: break

    # Process Phase
    paused, frame_idx, last_f = False, 0, None
    zone_active = len(zone_poly)>=3

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break
            last_f = frame.copy()
        else: frame = last_f.copy()

        if not paused:
            res = model.track(frame, conf=CONF, iou=IOU, persist=True, tracker=TRACKER, verbose=False, classes=CLASSES)
            if zone_active: _draw_zone(frame, zone_poly)
            
            if res and res[0].boxes is not None:
                for b in res[0].boxes:
                    xy = b.xyxy[0].cpu().numpy().astype(int)
                    tid = int(b.id[0]) if b.id is not None else -1
                    cx, cy = _centroid(xy)
                    inside = _inside_zone((cx,cy), zone_poly)
                    
                    cv2.rectangle(frame, (xy[0],xy[1]), (xy[2],xy[3]), _GRN if inside else (50,50,220), 1)
                    if DRAW_TRACKS and tid!=-1:
                        trails[tid].append((cx,cy))
                        pts = list(trails[tid])
                        for pi in range(1,len(pts)):
                            cv2.line(frame, pts[pi-1], pts[pi], (0,200,80) if inside else (40,40,200), 1)
                    
                    if tid!=-1:
                        if tid not in prev_sides: prev_sides[tid]={}
                        for li,ln in enumerate(counting_lines):
                            cur = _side((cx,cy), ln["p1"], ln["p2"])
                            prev = prev_sides[tid].get(li)
                            if prev is not None and prev!=0 and cur!=0 and (prev>0)!=(cur>0):
                                if inside:
                                    if count_mode=="both_add": ln["in" if prev>0 else "out"]+=1
                                    elif count_mode=="one_way" and prev>0: ln["in"]+=1
                                    elif count_mode=="net": ln["in" if prev>0 else "out"]+=1
                            if cur!=0: prev_sides[tid][li] = cur
                            
        _draw_lines(frame, counting_lines)
        _draw_hud(frame, counting_lines, count_mode, zone_active)
        if paused: _puttext(frame, "PAUSED", (fw//2-100, fh//2), 2.0, _ACC, 4)
        
        if writer: writer.write(frame)
        cv2.imshow(WIN, frame)
        k = cv2.waitKey(1)&0xFF
        if k in (ord('q'),27): break
        elif k==ord(' '): paused = not paused
        elif k==ord('r'):
            for ln in counting_lines: ln["in"]=ln["out"]=0
            prev_sides.clear()

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
