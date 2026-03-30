"""
CV-Count — Interactive Line-Crossing People Counter
====================================================
FLOW:
  1. On-screen settings window  → click to pick model, counting mode, save
  2. Setup on first frame       → draw ZONE (Z) and/or LINES (N), then SPACE
  3. Live processing            → YOLO counts people crossing each line

ZONE: Polygon drawn with Z + multiple clicks (Enter/Z to close).
  Green bbox  = person inside zone  → counted
  Red bbox    = person outside zone → ignored

Controls during setup & processing:
    N  → draw counting line (click 2 pts)
    Z  → draw detection zone (click polygon pts, Enter/Z to close)
    X  → clear zone
    C  → clear all lines + zone + reset counts
    R  → reset counts only
    Space  → start (setup) / pause-resume (processing)
    Q / Esc → quit
"""

# ─────────────────────── CONFIG ──────────────────────────────────────────────

VIDEO_PATH     = "01.50fps.mp4.mp4"
MODELS_DIR     = "models"
CLASSES        = [0]          # 0=person; []=all
CONF           = 0.35
IOU            = 0.45
TRACKER        = "bytetrack.yaml"
DRAW_TRACKS    = True
TRAIL_LEN      = 30
SKIP_FRAMES    = 0
LINE_THICKNESS = 3
HUD_ALPHA      = 0.60
MAX_WIN_W      = 1280
MAX_WIN_H      = 800

# ─────────────────────────────────────────────────────────────────────────────

import sys, os, collections
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] Run:  C:\\Python314\\python.exe -m pip install -r requirements.txt")

try:
    import lap  # noqa: F401
except ImportError:
    sys.exit("[ERROR] Run:  C:\\Python314\\python.exe -m pip install lap")


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
    """True if pt is inside the closed polygon (or no zone set)."""
    if len(zone_poly) < 3:
        return True
    poly = np.array(zone_poly, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(pt[0]),float(pt[1])), False) >= 0


# ─────────────────────── drawing helpers ─────────────────────────────────────

def _draw_zone(img, zone_poly, in_progress=False, cursor=None):
    """Render the detection zone polygon."""
    if not zone_poly:
        return
    pts = np.array(zone_poly, dtype=np.int32)
    closed = not in_progress          # closed polygon when not drawing

    if closed and len(pts) >= 3:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0,80,20))
        cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
        cv2.polylines(img, [pts], True, (0,220,80), 2, cv2.LINE_AA)
    else:
        # draw in-progress lines
        for i in range(len(pts)-1):
            cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), (0,180,80), 2, cv2.LINE_AA)
        if cursor and len(pts) > 0:
            cv2.line(img, tuple(pts[-1]), cursor, (0,180,80), 1, cv2.LINE_AA)
        if len(pts) >= 3:
            # preview close edge
            cv2.line(img, tuple(pts[-1]), tuple(pts[0]), (0,100,50), 1, cv2.LINE_AA)

    # draw corner dots
    for p in zone_poly:
        cv2.circle(img, p, 5, (0,255,100), -1)
        cv2.circle(img, p, 5, (255,255,255), 1)

    if in_progress and len(zone_poly) >= 3:
        msg = "Enter/Z to close zone"
        _puttext(img, msg, (zone_poly[0][0]+8, zone_poly[0][1]-10),
                 0.45, (0,255,100))


def _draw_lines_on(frame, counting_lines):
    for i, ln in enumerate(counting_lines):
        a,b = ln["p1"],ln["p2"]
        cv2.line(frame,a,b,ln["color"],LINE_THICKNESS,cv2.LINE_AA)
        mid = ((a[0]+b[0])//2,(a[1]+b[1])//2)
        dx,dy = b[0]-a[0],b[1]-a[1]
        L = max((dx**2+dy**2)**0.5,1)
        nx,ny = int(-dy/L*18),int(dx/L*18)
        cv2.arrowedLine(frame,mid,(mid[0]+nx,mid[1]+ny),(255,255,255),1,tipLength=0.4)
        _puttext(frame,f"L{i+1}",(mid[0]+8,mid[1]-6),0.55,ln["color"],2)


def _draw_hud(frame, counting_lines, count_mode, zone_active):
    pad,row_h,margin = 10,30,10
    extra = 1 if zone_active else 0
    rows  = 1 + len(counting_lines) + extra + 1 + 1
    ph    = rows*row_h + 2*pad
    pw    = 250
    x0,y0 = margin,margin

    ov = frame.copy()
    cv2.rectangle(ov,(x0,y0),(x0+pw,y0+ph),(15,15,15),-1)
    cv2.addWeighted(ov,HUD_ALPHA,frame,1-HUD_ALPHA,0,frame)

    ml = {"both_add":"BOTH+1","one_way":"ONE-WAY","net":"NET+/-"}[count_mode]
    _puttext(frame,f"CV-COUNT [{ml}]",(x0+pad,y0+pad+14),0.40,(160,160,160))

    yc = y0+pad+row_h
    if zone_active:
        _puttext(frame,"Zone active",(x0+pad,yc+9),0.40,(0,200,80))
        yc += row_h

    grand=0
    for i,ln in enumerate(counting_lines):
        col = ln["color"]
        if count_mode=="both_add":  val=ln["in"]+ln["out"]; txt=str(val)
        elif count_mode=="one_way": val=ln["in"];            txt=str(val)
        else:                       val=ln["in"]-ln["out"]; txt=f"{val:+d}" if val else "0"
        grand+=val
        cv2.circle(frame,(x0+pad+8,yc+4),7,col,-1)
        _puttext(frame,f"Line {i+1}:",(x0+pad+22,yc+9),0.46,(200,200,200))
        sub = f"(dn{ln['in']} up{ln['out']})" if count_mode!="one_way" else f"({ln['in']})"
        _puttext(frame,sub,(x0+pad+22,yc+21),0.33,(120,120,120))
        vx=x0+pw-pad-max(len(txt)*11,20)
        _puttext(frame,txt,(vx,yc+9),0.55,col)
        yc+=row_h

    cv2.line(frame,(x0+pad,yc),(x0+pw-pad,yc),(70,70,70),1)
    yc+=row_h-8
    _puttext(frame,"TOTAL:",(x0+pad,yc+9),0.50,(220,220,220))
    g=f"{grand:+d}" if count_mode=="net" and grand else str(grand)
    gx=x0+pw-pad-max(len(g)*14,20)
    cv2.putText(frame,g,(gx,yc+9),cv2.FONT_HERSHEY_DUPLEX,0.65,(0,255,180),2,cv2.LINE_AA)


# ─────────────────────── on-screen settings ──────────────────────────────────

_MODELS = [
    ("YOLO26 Nano   (fastest, NMS-free)","yolo26n.pt"),
    ("YOLO26 Small  (fast + accurate)",  "yolo26s.pt"),
    ("YOLO26 Medium (balanced)",         "yolo26m.pt"),
    ("YOLO26 Large  (most accurate)",    "yolo26l.pt"),
    ("YOLO11 Nano   (legacy, ~39 MB)",   "yolo11n.pt"),
    ("YOLO11 Medium (legacy, ~170 MB)",  "yolo11m.pt"),
    ("YOLOv8 Nano   (legacy, ~6 MB)",    "yolov8n.pt"),
    ("YOLOv8 Medium (legacy, ~52 MB)",   "yolov8m.pt"),
]
_MODES = [
    ("Both directions  --  +1 either way",      "both_add"),
    ("One-way only     --  +1 for IN side only", "one_way"),
    ("Net occupancy    --  IN +1 / OUT -1",      "net"),
]
_LATEST = {"yolo26n.pt","yolo26s.pt","yolo26m.pt","yolo26l.pt"}

_BG=(18,18,22); _CARD=(28,28,36); _ACC=(0,200,255); _GRN=(0,220,110)
_SEL=(0,60,90); _HOV=(35,35,48); _TXT=(210,210,220); _MUT=(110,110,128)

def _mdl_cached(fn):
    return os.path.isfile(os.path.join(MODELS_DIR,fn)) or (os.path.isabs(fn) and os.path.isfile(fn))


def settings_screen():
    SW,SH = 860,660
    WIN   = "CV-Count -- Settings"
    cv2.namedWindow(WIN,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN,SW,SH)

    st = {"model":2,"mode":0,"save":True,"start":False}
    mouse = {"x":0,"y":0,"dn":False}

    def on_mouse(ev,x,y,fl,_):
        mouse["x"]=x; mouse["y"]=y
        if ev==cv2.EVENT_LBUTTONDOWN: mouse["dn"]=True
    cv2.setMouseCallback(WIN,on_mouse)

    RH=38; PAD=18; CW=SW-2*PAD
    SMY = 68
    SMoY= SMY+26+len(_MODELS)*RH+16
    SSY = SMoY+26+len(_MODES)*RH+16
    STBY= SSY+70

    def _row(img,i,sel,hov,label,tag=None):
        ry = (SMY if tag is not None else SMoY)+26+i*RH
        if tag is None: ry = SMoY+26+i*RH   # mode rows
        selected=i==sel; hovered=i==hov and not selected
        bg=_SEL if selected else (_HOV if hovered else _BG)
        cv2.rectangle(img,(PAD,ry),(PAD+CW,ry+RH-2),bg,-1)
        bx,bc=PAD+16,ry+RH//2
        if selected:
            cv2.circle(img,(bx,bc),8,_ACC,-1); cv2.circle(img,(bx,bc),4,_BG,-1)
        else:
            cv2.circle(img,(bx,bc),8,_MUT,1)
        _puttext(img,label,(PAD+34,ry+RH//2+6),0.50,(255,255,255) if selected else _TXT)
        if tag:
            tx=PAD+34+len(label)*9+8
            if tag=="dl":
                cv2.rectangle(img,(tx,ry+9),(tx+60,ry+RH-10),(0,60,30),-1)
                _puttext(img,"CACHED",(tx+4,ry+RH-12),0.34,_GRN)
            elif tag=="new":
                cv2.rectangle(img,(tx,ry+9),(tx+60,ry+RH-10),(0,40,70),-1)
                _puttext(img,"LATEST",(tx+4,ry+RH-12),0.34,_ACC)

    def render(hm,hmo):
        img=np.full((SH,SW,3),_BG,dtype=np.uint8)
        cv2.rectangle(img,(0,0),(SW,54),_CARD,-1)
        cv2.line(img,(0,54),(SW,54),_ACC,1)
        _puttext(img,"CV-COUNT",(PAD,34),0.80,_ACC,bold=True)
        _puttext(img,"Line-Crossing Counter  --  Setup",(PAD+118,34),0.46,_MUT)

        def _sec(y,lbl):
            cv2.rectangle(img,(PAD,y),(PAD+CW,y+22),_CARD,-1)
            _puttext(img,lbl,(PAD+8,y+16),0.46,_ACC,bold=True)

        # MODEL
        _sec(SMY,"  MODEL")
        for i,(lbl,fn) in enumerate(_MODELS):
            ry=SMY+26+i*RH
            selected=i==st["model"]; hovered=i==hm and not selected
            bg=_SEL if selected else (_HOV if hovered else _BG)
            cv2.rectangle(img,(PAD,ry),(PAD+CW,ry+RH-2),bg,-1)
            bx,bc=PAD+16,ry+RH//2
            if selected: cv2.circle(img,(bx,bc),8,_ACC,-1); cv2.circle(img,(bx,bc),4,_BG,-1)
            else:        cv2.circle(img,(bx,bc),8,_MUT,1)
            _puttext(img,lbl,(PAD+34,ry+RH//2+6),0.50,(255,255,255) if selected else _TXT)
            tx=PAD+34+len(lbl)*9+8
            if _mdl_cached(fn):
                cv2.rectangle(img,(tx,ry+9),(tx+62,ry+RH-10),(0,60,30),-1)
                _puttext(img,"CACHED",(tx+4,ry+RH-12),0.34,_GRN)
            elif fn in _LATEST:
                cv2.rectangle(img,(tx,ry+9),(tx+62,ry+RH-10),(0,40,70),-1)
                _puttext(img,"LATEST",(tx+4,ry+RH-12),0.34,_ACC)

        # MODE
        _sec(SMoY,"  COUNTING MODE")
        for i,(lbl,_) in enumerate(_MODES):
            ry=SMoY+26+i*RH
            selected=i==st["mode"]; hovered=i==hmo and not selected
            bg=_SEL if selected else (_HOV if hovered else _BG)
            cv2.rectangle(img,(PAD,ry),(PAD+CW,ry+RH-2),bg,-1)
            bx,bc=PAD+16,ry+RH//2
            if selected: cv2.circle(img,(bx,bc),8,_ACC,-1); cv2.circle(img,(bx,bc),4,_BG,-1)
            else:        cv2.circle(img,(bx,bc),8,_MUT,1)
            _puttext(img,lbl,(PAD+34,ry+RH//2+6),0.50,(255,255,255) if selected else _TXT)

        # SAVE
        _sec(SSY,"  SAVE OUTPUT VIDEO")
        yb=SSY+28
        yc=_SEL if st["save"] else _CARD
        nc=(50,30,60) if not st["save"] else _CARD
        cv2.rectangle(img,(PAD+8,yb),(PAD+118,yb+32),yc,-1)
        cv2.rectangle(img,(PAD+8,yb),(PAD+118,yb+32),_ACC if st["save"] else _MUT,1)
        _puttext(img,"YES",(PAD+42,yb+23),0.55,(0,220,100) if st["save"] else _MUT,bold=st["save"])
        cv2.rectangle(img,(PAD+128,yb),(PAD+238,yb+32),nc,-1)
        cv2.rectangle(img,(PAD+128,yb),(PAD+238,yb+32),_MUT,1)
        _puttext(img,"NO",(PAD+168,yb+23),0.55,(180,80,180) if not st["save"] else _MUT,bold=not st["save"])
        if st["save"]:
            base=os.path.splitext(os.path.basename(VIDEO_PATH if not isinstance(VIDEO_PATH,int) else "webcam"))[0]
            _puttext(img,f"Output: {base}_counted.mp4",(PAD+252,yb+23),0.40,_MUT)

        # START
        hs=_in_rect(mouse["x"],mouse["y"],SW//2-120,STBY,240,54)
        cv2.rectangle(img,(SW//2-120,STBY),(SW//2+120,STBY+54),(0,170,90) if hs else (0,140,70),-1)
        cv2.rectangle(img,(SW//2-120,STBY),(SW//2+120,STBY+54),_GRN,1)
        _puttext(img,"START",(SW//2-38,STBY+36),0.85,(255,255,255),bold=True)

        _puttext(img,"Click to select  |  Q / Esc to quit",(PAD,SH-10),0.37,_MUT)
        return img

    hm=hmo=-1
    while True:
        mx,my=mouse["x"],mouse["y"]
        hm=-1
        for i in range(len(_MODELS)):
            if _in_rect(mx,my,PAD,SMY+26+i*RH,CW,RH-2): hm=i; break
        hmo=-1
        for i in range(len(_MODES)):
            if _in_rect(mx,my,PAD,SMoY+26+i*RH,CW,RH-2): hmo=i; break

        if mouse["dn"]:
            mouse["dn"]=False
            if hm>=0: st["model"]=hm
            elif hmo>=0: st["mode"]=hmo
            elif _in_rect(mx,my,PAD+8,SSY+28,110,32):  st["save"]=True
            elif _in_rect(mx,my,PAD+128,SSY+28,110,32): st["save"]=False
            elif _in_rect(mx,my,SW//2-120,STBY,240,54): st["start"]=True

        cv2.imshow(WIN,render(hm,hmo))
        key=cv2.waitKey(16)&0xFF
        if key in (ord('q'),27): cv2.destroyAllWindows(); sys.exit(0)
        if st["start"]: break

    cv2.destroyWindow(WIN)
    mdl=_MODELS[st["model"]][1]
    mode=_MODES[st["mode"]][1]
    opath=None
    if st["save"]:
        base=os.path.splitext(os.path.basename(VIDEO_PATH if not isinstance(VIDEO_PATH,int) else "webcam"))[0]
        opath=f"{base}_counted.mp4"
    print(f"[INFO] Model:{mdl}  Mode:{mode}  Save:{opath or 'No'}")
    return mdl,mode,opath


# ─────────────────────── main ─────────────────────────────────────────────────

def main():
    model_name, count_mode, output_path = settings_screen()

    # load model
    os.makedirs(MODELS_DIR,exist_ok=True)
    mp=model_name if os.path.isabs(model_name) else os.path.join(MODELS_DIR,model_name)
    print(f"[INFO] Loading model: {mp} ...")
    _orig=os.getcwd(); os.chdir(MODELS_DIR)
    model=YOLO(os.path.basename(mp))
    os.chdir(_orig)
    print("[INFO] Model ready.\n")
    print("N=line | Z=zone | X=clear zone | C=clear all | R=reset | Space=start/pause | Q=quit")

    # open video
    cap=cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): sys.exit(f"[ERROR] Cannot open: {VIDEO_PATH}")
    fw=int(cap.get(3)); fh=int(cap.get(4))
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    totf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {fw}x{fh} @ {fps:.1f}fps  ({totf} frames)")

    scale=min(MAX_WIN_W/fw,MAX_WIN_H/fh,1.0)
    dw,dh=int(fw*scale),int(fh*scale)

    writer=None
    if output_path:
        writer=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(fw,fh))
        print(f"[INFO] Saving -> {output_path}")

    # shared state
    counting_lines=[]
    zone_poly=[]       # list of (x,y) tuples; empty = no zone
    zone_mode=False    # actively drawing zone polygon
    prev_sides={}
    trails=collections.defaultdict(lambda: collections.deque(maxlen=TRAIL_LEN))
    draw_mode=False; draw_start=None

    WIN="CV-Count  [N=line|Z=zone|X=clr zone|C=clear|R=reset|Space=start/pause|Q=quit]"
    cv2.namedWindow(WIN,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN,dw,dh)

    def on_mouse(ev,x,y,flags,_):
        nonlocal draw_mode,draw_start,zone_mode,zone_poly
        if ev==cv2.EVENT_LBUTTONDOWN:
            if zone_mode:
                zone_poly.append((x,y))
                print(f"[ZONE] Point {len(zone_poly)}: ({x},{y})")
            elif draw_mode:
                if draw_start is None:
                    draw_start=(x,y); print(f"[LINE] P1={draw_start} -- click P2")
                else:
                    col=_PALETTE[len(counting_lines)%len(_PALETTE)]
                    counting_lines.append({"p1":draw_start,"p2":(x,y),"color":col,"in":0,"out":0})
                    print(f"[LINE] Line {len(counting_lines)}: {draw_start} -> ({x},{y})")
                    draw_start=None; draw_mode=False
        elif ev==cv2.EVENT_MOUSEMOVE:
            pass  # tracked via mouse["x/y"] if needed

    cv2.setMouseCallback(WIN,on_mouse)
    mouse_pos=[0,0]

    def on_mouse2(ev,x,y,flags,_):
        mouse_pos[0]=x; mouse_pos[1]=y
        on_mouse(ev,x,y,flags,_)
    cv2.setMouseCallback(WIN,on_mouse2)

    # ── read first frame for setup
    ret,first_frame=cap.read()
    if not ret: sys.exit("[ERROR] Cannot read first frame.")
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    print("[SETUP] Draw zone (Z) and/or lines (N), then SPACE to start.")

    # ═══════════ PHASE 1 — Setup ═════════════════════════════════════════════
    while True:
        sf=first_frame.copy()

        # draw zone
        cursor=tuple(mouse_pos) if zone_mode else None
        _draw_zone(sf,zone_poly,in_progress=zone_mode,cursor=cursor)
        _draw_lines_on(sf,counting_lines)

        # line draw helpers
        if draw_mode and draw_start:
            cv2.circle(sf,draw_start,6,(0,255,255),-1)
            _puttext(sf,"Click P2",(draw_start[0]+10,draw_start[1]-10),0.55,(0,255,255))
        elif draw_mode:
            _puttext(sf,"Click Point 1 (start)",(20,fh-20),0.60,(0,255,255),2)

        # top banner
        if zone_mode:
            msg="  Click to add zone points -- Enter or Z to close  "
            bc=(0,150,50)
        elif draw_mode:
            msg="  Click 2 points to draw counting line  "
            bc=(0,100,150)
        else:
            msg="  N=line  Z=zone  SPACE=start  "
            bc=(15,15,15)
        (tw,_),_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.62,2)
        bx,by=(fw-tw)//2,38
        cv2.rectangle(sf,(bx-12,by-24),(bx+tw+12,by+10),bc,-1)
        cv2.rectangle(sf,(bx-12,by-24),(bx+tw+12,by+10),(0,200,255),1)
        _puttext(sf,msg,(bx,by),0.62,(0,220,255),2)

        info_parts=[]
        if zone_poly: info_parts.append(f"Zone: {len(zone_poly)} pts {'(open)' if zone_mode else '(closed)'}")
        if counting_lines: info_parts.append(f"{len(counting_lines)} line(s)")
        if info_parts:
            _puttext(sf,"  |  ".join(info_parts)+" -- SPACE to start",
                     (12,fh-12),0.48,(0,220,100))

        cv2.imshow(WIN,sf)
        key=cv2.waitKey(30)&0xFF

        if key in (ord('q'),27): cap.release(); cv2.destroyAllWindows(); return
        elif key==ord('n'):
            zone_mode=False; draw_mode=True; draw_start=None
            print("[LINE] Draw mode ON -- click 2 points.")
        elif key==ord('z'):    # toggle zone mode / close zone
            if zone_mode:
                if len(zone_poly)>=3:
                    zone_mode=False; print(f"[ZONE] Closed with {len(zone_poly)} points.")
                else:
                    print("[ZONE] Need at least 3 points to close zone.")
            else:
                draw_mode=False; draw_start=None; zone_mode=True
                print("[ZONE] Zone mode ON -- click polygon vertices.")
        elif key==ord('x'):
            zone_poly.clear(); zone_mode=False; print("[ZONE] Zone cleared.")
        elif key in (13,10):   # Enter -- close zone
            if zone_mode and len(zone_poly)>=3:
                zone_mode=False; print(f"[ZONE] Closed with {len(zone_poly)} points.")
        elif key==ord('c'):
            counting_lines.clear(); zone_poly.clear()
            draw_mode=False; draw_start=None; zone_mode=False
            print("[INFO] Cleared all.")
        elif key==ord(' '):
            if not counting_lines:
                print("[WARN] Draw at least one line first (press N).")
            else:
                if zone_mode and len(zone_poly)>=3:
                    zone_mode=False
                    print(f"[ZONE] Auto-closed with {len(zone_poly)} points.")
                print(f"[INFO] Starting -- {len(counting_lines)} line(s)"
                      + (f", zone active ({len(zone_poly)} pts)" if len(zone_poly)>=3 else ""))
                break

    zone_active = len(zone_poly) >= 3

    # ═══════════ PHASE 2 — Process ═══════════════════════════════════════════
    paused=False; frame_idx=0; last_frame=None

    while True:
        if not paused:
            for _ in range(max(1,SKIP_FRAMES+1)):
                ret,frame=cap.read()
                if not ret: frame=None; break
                frame_idx+=1
            if frame is None: print("[INFO] End of video."); break
            last_frame=frame.copy()
        else:
            frame=last_frame.copy() if last_frame is not None \
                  else np.zeros((fh,fw,3),dtype=np.uint8)

        # inference
        if not paused:
            kw=dict(conf=CONF,iou=IOU,persist=True,tracker=TRACKER,verbose=False)
            if CLASSES: kw["classes"]=CLASSES
            results=model.track(frame,**kw)
        else:
            results=None

        # draw zone (behind detections)
        if zone_active:
            _draw_zone(frame,zone_poly,in_progress=False)

        # detections
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                tid  = int(box.id[0]) if box.id is not None else -1
                cx,cy = _centroid(xyxy)

                in_zone = _inside_zone((cx,cy), zone_poly)

                # bbox color: green=inside/counted, red=outside/ignored
                bbox_col = (50,200,50) if in_zone else (40,40,220)
                cv2.rectangle(frame,(xyxy[0],xyxy[1]),(xyxy[2],xyxy[3]),bbox_col,1)
                if tid!=-1:
                    _puttext(frame,f"#{tid}",(xyxy[0]+4,xyxy[1]-6),0.42,
                             (50,255,100) if in_zone else (80,80,255))

                # trails
                if DRAW_TRACKS and tid!=-1:
                    trails[tid].append((cx,cy))
                    pts=list(trails[tid])
                    trail_col=(50,200,50) if in_zone else (40,40,200)
                    for pi in range(1,len(pts)):
                        a2=pi/len(pts)
                        tc=tuple(int(c*a2) for c in trail_col)
                        cv2.line(frame,pts[pi-1],pts[pi],tc,1,cv2.LINE_AA)

                # crossing (only if in zone)
                if tid==-1: continue
                if tid not in prev_sides: prev_sides[tid]={}
                for li,ln in enumerate(counting_lines):
                    cur=_side((cx,cy),ln["p1"],ln["p2"])
                    prev=prev_sides[tid].get(li)
                    if prev is not None and prev!=0 and cur!=0 and (prev>0)!=(cur>0):
                        if in_zone:   # only count when inside zone
                            going_in=(prev>0)
                            if count_mode=="both_add":
                                counting_lines[li]["in" if going_in else "out"]+=1
                            elif count_mode=="one_way":
                                if going_in: counting_lines[li]["in"]+=1
                            else:
                                counting_lines[li]["in" if going_in else "out"]+=1
                    if cur!=0: prev_sides[tid][li]=cur

        _draw_lines_on(frame,counting_lines)
        _draw_hud(frame,counting_lines,count_mode,zone_active)

        # line draw feedback
        if draw_mode and draw_start:
            cv2.circle(frame,draw_start,6,(0,255,255),-1)
            _puttext(frame,"Click P2",(draw_start[0]+10,draw_start[1]-10),0.55,(0,255,255))
        elif draw_mode:
            _puttext(frame,"Click Point 1",(20,fh-20),0.60,(0,255,255),2)

        # progress bar
        if totf>0 and not paused:
            cv2.rectangle(frame,(0,fh-4),(int(fw*frame_idx/totf),fh),(0,200,255),-1)

        if paused:
            cv2.putText(frame,"PAUSED",(fw//2-70,fh//2),cv2.FONT_HERSHEY_DUPLEX,
                        1.4,(0,200,255),3,cv2.LINE_AA)

        if writer: writer.write(frame)
        cv2.imshow(WIN,frame)

        key=cv2.waitKey(1)&0xFF
        if key in (ord('q'),27): print("[INFO] Quit."); break
        elif key==ord('n'): draw_mode=True; draw_start=None; print("[LINE] Draw mode ON")
        elif key==ord('c'):
            counting_lines.clear(); prev_sides.clear(); trails.clear()
            zone_poly.clear(); zone_active=False
            draw_mode=False; draw_start=None; print("[INFO] Cleared all.")
        elif key==ord('r'):
            for ln in counting_lines: ln["in"]=ln["out"]=0
            prev_sides.clear(); print("[INFO] Counts reset.")
        elif key==ord(' '): paused=not paused; print(f"[INFO] {'Paused' if paused else 'Resumed'}.")

    cap.release()
    if writer: writer.release(); print(f"[INFO] Saved: {output_path}")

    print("\n"+"="*44+"\n  FINAL COUNTS  ["+count_mode+"]\n"+"="*44)
    grand=0
    for i,ln in enumerate(counting_lines):
        if count_mode=="both_add":   v=ln["in"]+ln["out"]; info=f"(dn{ln['in']} up{ln['out']})"
        elif count_mode=="one_way":  v=ln["in"]; info=""
        else:                        v=ln["in"]-ln["out"]; info=f"(in={ln['in']} out={ln['out']})"
        grand+=v; print(f"  Line {i+1}: {v:>5}  {info}")
    print(f"  {'─'*34}\n  TOTAL  : {grand}\n"+"="*44)
    cv2.destroyAllWindows()


if __name__=="__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
