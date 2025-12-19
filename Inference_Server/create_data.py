import cv2
import os
import numpy as np
from ultralytics import YOLO

# ==============================
# 설정
# ==============================
VIDEO_PATH = r"input_video.mp4"
YOLO_WEIGHTS = "yolo11n-pose.pt"

OUT_IMG_DIR = "dataset/images"
OUT_LABEL_DIR = "dataset/labels"

CONF_THRES = 0.5
IOU_THRES = 0.5
FRAME_SKIP = 1        # 1 = 모든 프레임 사용
MAX_PERSONS = 1       # 수면이므로 1명만 사용

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# ==============================
# 모델 로드
# ==============================
model = YOLO(YOLO_WEIGHTS)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ 비디오를 열 수 없습니다.")

frame_idx = 0
saved_count = 0
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

# ==============================
# 프레임 루프
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        continue

    h, w = frame.shape[:2]

    results = model(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
    result = results[0]

    if len(result.boxes) == 0 or result.keypoints is None:
        continue

    # ------------------------------
    # 가장 신뢰도 높은 1명만 사용
    # ------------------------------
    box = result.boxes.xyxy[0].cpu().numpy()
    box_n = result.boxes.xyxyn[0].cpu().numpy()

    kpts_xy = result.keypoints.xyn[0].cpu().numpy()     # (17,2)
    kpts_conf = result.keypoints.conf[0].cpu().numpy()  # (17,)

    # ------------------------------
    # YOLO Pose 라벨 구성
    # ------------------------------
    cls_id = 0  # 임시 클래스 (나중에 posture class로 재라벨링)

    label_data = [cls_id]
    label_data.extend(box_n.tolist())  # xc yc w h

    for (x, y), c in zip(kpts_xy, kpts_conf):
        label_data.extend([x, y, c])

    label_str = " ".join(f"{v:.6f}" for v in label_data)

    # ------------------------------
    # 저장
    # ------------------------------
    fname = f"{video_name}_{saved_count:06d}"

    img_path = os.path.join(OUT_IMG_DIR, f"{fname}.jpg")
    label_path = os.path.join(OUT_LABEL_DIR, f"{fname}.txt")

    cv2.imwrite(img_path, frame)

    with open(label_path, "w") as f:
        f.write(label_str + "\n")

    saved_count += 1

cap.release()

print(f"✅ 완료: {saved_count} 프레임 저장됨")
