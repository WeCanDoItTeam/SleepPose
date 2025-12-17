import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta

# =========================
# 설정값
# =========================
video_path = "downloads/TEST_0.mp4"
model_path = "pose_pt/pose_011/weights/best.pt"

IMG_SIZE = 640
FRAME_SKIP = 1
MIN_FRAMES = 1
CONF_THRES = 0.5
IOU_THRES = 0.5

# =========================
# 모델 & 영상 로드
# =========================
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

model = YOLO(model_path)
names = model.names

# 🔴 영상 시작 datetime (기준 시각)
video_start_time = datetime.now()

# =========================
# 상태 변수
# =========================
frame_count = 0

current_class = None
start_frame = None
count_frames = 0

pose_segments = []

# =========================
# 비디오 처리 루프
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    input_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    detected_class = None

    if frame_count % FRAME_SKIP == 0:
        results = model(
            input_frame,
            imgsz=IMG_SIZE,
            device=0,
            half=True,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False
        )

        for r in results:
            if r.boxes is not None and len(r.boxes.cls) > 0:
                detected_class = int(r.boxes.cls[0])

                boxes = r.boxes.xyxy.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy()

                for box, cls_id in zip(boxes, cls_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = names[int(cls_id)]

                    cv2.rectangle(input_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        input_frame,
                        class_name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

            # =========================
            # 키포인트 그리기
            # =========================
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                keypoints = r.keypoints.xy.cpu().numpy()
                for person_kps in keypoints:
                    for x, y in person_kps:
                        x, y = int(x), int(y)
                        cv2.circle(input_frame, (x, y), 3, (0, 0, 255), -1)


    # =========================
    # 자세 구간 추적 로직
    # =========================
    if detected_class == current_class:
        count_frames += 1
    else:
        if current_class is not None and count_frames >= MIN_FRAMES:
            end_frame = frame_count - 1

            pose_segments.append({
                "class_id": current_class,
                "class_name": names[current_class],
                "start_time": video_start_time + timedelta(seconds=start_frame / fps),
                "end_time": video_start_time + timedelta(seconds=end_frame / fps)
            })

        current_class = detected_class
        start_frame = frame_count
        count_frames = 1

    cv2.imshow("Sleep Pose Detection", input_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# 마지막 구간 처리
# =========================
if current_class is not None and count_frames >= MIN_FRAMES:
    pose_segments.append({
        "class_id": current_class,
        "class_name": names[current_class],
        "start_time": video_start_time + timedelta(seconds=start_frame / fps),
        "end_time": video_start_time + timedelta(seconds=frame_count / fps)
    })

cap.release()
cv2.destroyAllWindows()

# =========================
# 결과 출력
# =========================
print("\n===== Pose Time Segments =====")
for seg in pose_segments:
    print(
        f"[{seg['class_name']}] "
        f"{seg['start_time'].strftime('%Y-%m-%d %H:%M:%S')} ~ "
        f"{seg['end_time'].strftime('%Y-%m-%d %H:%M:%S')}"
    )



# import cv2
# import numpy as np
# from ultralytics import YOLO

# video_path = "downloads/TEST_3.mp4"

# # 영상 불러오기
# cap = cv2.VideoCapture(video_path)

# # YOLO Pose 모델 불러오기
# model = YOLO("pt/pose_011/weights/best.pt")
# names = model.names  # 클래스 ID → 이름

# FRAME_SKIP = 1
# frame_count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     input_frame = cv2.resize(frame, (640, 640))
    
#     frame_count += 1

#     if frame_count % FRAME_SKIP == 0:
#         results = model(
#             input_frame,
#             imgsz=640,
#             device=0,   # GPU 사용
#             half=True,  # FP16
#             verbose=False,
#             conf=0.5,
#             iou=0.4
#         )

#         for r in results:
#             print("Class IDs:", r.boxes.cls)
#             if r.boxes is not None:
#                 boxes = r.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
#                 cls_ids = r.boxes.cls.cpu().numpy()  # 클래스 ID
#                 for box, cls_id in zip(boxes, cls_ids):
#                     x1, y1, x2, y2 = box.astype(int)
#                     class_name = names[int(cls_id)]

#                     # 바운딩 박스 그리기
#                     cv2.rectangle(input_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     # 클래스명 표시
#                     cv2.putText(
#                         input_frame,
#                         class_name,
#                         (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.8,
#                         (0, 255, 0),
#                         2
#                     )
        
#     cv2.imshow("Sleep Pose Detection", input_frame)

#     # ESC 키를 누르면 종료
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


# # 클래스 아이디, 타임스탬프
# # 시작 시간, 종료 시간
# # n 프레임 이상 이어질 시 시작 시간과 종료 시 끝시간, 클래스 아이디.
# # => 각 자세의 시간을 배열로 저장