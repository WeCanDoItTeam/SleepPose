from ultralytics import YOLO
import cv2
import numpy as np
import os

# =============================
# 설정
# =============================

# 어깨가 어느정도 수평이면 - 얼굴 신뢰도 확인하고 엎드린거(<0.4) 바로누운거(0.4<) 손목이 위에 있으면 핸드업 반환
laying = "laying" #0 얼굴 0.5 이상
side_l = "side_l" #1 0.4 미만
side_r = "side_r" #1
hand_up = "hand_up" #2
back = "back" #3 얼굴 0.4 미만
sit = "sit" #4 
stand = "stand" #4

NAME = stand

SAVE_ROOT = r"C:\Users\USER\Documents\Github\SleepPose\Inference_Server\data\Lee"
IMG_DIR = os.path.join(SAVE_ROOT, "train", "images", "others")
LBL_DIR = os.path.join(SAVE_ROOT, "train", "labels", "others")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

FRAME_STRIDE = 5   # 15fps → 3fps
frame_idx = 0

VIDEO_PATH = rf"C:\Users\USER\Documents\Github\SleepPose\Inference_Server\lee_video\{NAME}.mp4"
YOLO_WEIGHTS = "yolo11n-pose.pt"
SAVE_PATH = rf"C:\Users\USER\Documents\Github\SleepPose\Inference_Server\infer_video\{NAME}.mp4"

CONF_THRES = 0.3
IOU_THRES = 0.5
KPT_ALPHA = 0.85

CLS_ID = 4

# COCO skeleton
SKELETON = [
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

# 얼굴 keypoint (COCO)
FACE_IDS = {
    0: "nose",
    1: "L_eye",
    2: "R_eye",
    3: "L_ear",
    4: "R_ear"
}


# =============================
# 유틸
# =============================
def to_yolo_pose_format(bbox, kpts, kpts_conf, img_w, img_h, cls_id):
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    parts = [str(cls_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]

    for i in range(17):
        x, y = kpts[i]
        conf = kpts_conf[i]

        if np.isnan(x) or np.isnan(y):
            parts += ["0", "0", "0"]
        else:
            nx = x / img_w
            ny = y / img_h

            if conf < 0.3:
                parts += [f"{nx:.6f}", f"{ny:.6f}", "1"]
            else:
                parts += [f"{nx:.6f}", f"{ny:.6f}", "2"]

    return " ".join(parts)

def draw_face_conf(frame, kpts_xy, kpts_conf):
    for idx, name in FACE_IDS.items():
        x, y = kpts_xy[idx]
        conf = kpts_conf[idx]

        if np.isnan(x):
            continue

        # 신뢰도별 색상
        if conf >= 0.5:
            color = (0, 255, 0)      # green
        elif conf >= 0.3:
            color = (0, 255, 255)    # yellow
        else:
            color = (0, 0, 255)      # red

        cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        cv2.putText(
            frame,
            f"{name}:{conf:.2f}",
            (int(x)+5, int(y)-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA
        )

# 적외선 환경 더 잘 보이게 처리
def ir_preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8,8))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# 스무스하게 키포인트 이동
def ema(prev, curr, alpha):
    return curr if prev is None else alpha * prev + (1 - alpha) * curr

# 급격한 좌우 뒤집힘 방지
def enforce_lr_consistency(kpts):
    """좌/우 스왑 방지"""
    pairs = [(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
    if np.isnan(kpts[5]).any() or np.isnan(kpts[6]).any():
        return kpts

    if kpts[5,0] > kpts[6,0]:
        for a,b in pairs:
            kpts[[a,b]] = kpts[[b,a]]
    return kpts


def to_bbox_norm(kpts, bbox):
    x1,y1,x2,y2 = bbox
    w,h = x2-x1, y2-y1
    k = kpts.copy()
    k[:,0] = (k[:,0] - x1) / w
    k[:,1] = (k[:,1] - y1) / h
    return k


def from_bbox_norm(kpts, bbox):
    x1,y1,x2,y2 = bbox
    w,h = x2-x1, y2-y1
    k = kpts.copy()
    k[:,0] = k[:,0] * w + x1
    k[:,1] = k[:,1] * h + y1
    return k


# =============================
# 모델 & 비디오
# =============================
model = YOLO(YOLO_WEIGHTS)
cap = cv2.VideoCapture(VIDEO_PATH)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

prev_kpts_norm = None

# =============================
# 추론 루프
# =============================
while True:
    ret, frame_ori = cap.read()
    if not ret:
        break

    frame = frame_ori.copy()

    proc = ir_preprocess(frame)
    r = model(proc, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]

    if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
        continue

    bbox = r.boxes.xyxy[0].cpu().numpy()
    kpts = r.keypoints.xy[0].cpu().numpy()      # (17,2)
    kpts_conf = r.keypoints.conf[0].cpu().numpy()  # (17,)


    # 좌우 고정
    kpts = enforce_lr_consistency(kpts)

    # bbox 기준 EMA
    k_norm = to_bbox_norm(kpts, bbox)
    k_norm = ema(prev_kpts_norm, k_norm, KPT_ALPHA)
    prev_kpts_norm = k_norm.copy()
    kpts = from_bbox_norm(k_norm, bbox)
    # 얼굴 키포인트 신뢰도 시각화
    draw_face_conf(frame, kpts, kpts_conf)


    # Draw bbox
    x1,y1,x2,y2 = bbox.astype(int)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    # Draw kpts
    for x,y in kpts:
        cv2.circle(frame,(int(x),int(y)),3,(0,0,255),-1)

    # Draw skeleton
    for a,b in SKELETON:
        cv2.line(frame,
                 tuple(kpts[a].astype(int)),
                 tuple(kpts[b].astype(int)),
                 (255,0,0),2)
        
    face_mean_conf = np.mean([kpts_conf[i] for i in FACE_IDS])
    cv2.putText(
        frame,
        f"Face mean conf: {face_mean_conf:.2f}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    if frame_idx % FRAME_STRIDE == 0:
        fname = f"{NAME}_frame_{frame_idx:06d}"

        # 이미지 저장
        img_path = os.path.join(IMG_DIR, fname + ".jpg")
        cv2.imwrite(img_path, frame_ori)

        # 라벨 저장
        label_str = to_yolo_pose_format(
            bbox=bbox,
            kpts=kpts,
            kpts_conf=kpts_conf,
            img_w=w,
            img_h=h,
            cls_id=CLS_ID
        )

        lbl_path = os.path.join(LBL_DIR, fname + ".txt")
        with open(lbl_path, "w") as f:
            f.write(label_str + "\n")

    frame_idx += 1

    cv2.imshow("YOLO Pose", cv2.resize(frame,(960,540)))

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
