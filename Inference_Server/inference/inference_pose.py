import subprocess
import numpy as np
import cv2
from ultralytics import YOLO

WIDTH, HEIGHT = 640, 640
FRAME_SKIP = 1 # 현재 스킵 안 함
FRAME_SIZE = WIDTH * HEIGHT * 3 #(RGB 3채널)

# ===== YOLO 모델 (전역 1회 로드) =====
model = YOLO("pose_pt/pose_03/weights/best.pt")
names = model.names


def run_ffmpeg_yolo(rtsp_url: str, ffmpeg_path: str, stop_flag: callable):
    """
    rtsp_url  : RTSP 주소
    ffmpeg_path : ffmpeg 실행 파일 경로
    stop_flag : 종료 여부를 반환하는 함수 (lambda)
    """

    cmd = [
        ffmpeg_path,
        "-rtsp_transport", "tcp",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-i", rtsp_url,
        "-vf", f"scale={WIDTH}:{HEIGHT}",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )

    frame_count = 0
    last_dets = None

    print("✅ FFmpeg YOLO 스트림 시작")

    try:
        while not stop_flag():
            raw_frame = process.stdout.read(FRAME_SIZE)
            if len(raw_frame) != FRAME_SIZE:
                print("❌ 스트림 끊김")
                break

            frame = (
                np.frombuffer(raw_frame, dtype=np.uint8)
                .reshape((HEIGHT, WIDTH, 3))
                .copy()
            )

            frame_count += 1

            # ===== YOLO 추론 =====
            if frame_count % FRAME_SKIP == 0:
                results = model(
                    frame,
                    imgsz=640,
                    device=0,
                    half=True,
                    verbose=False
                )

                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    clss = results[0].boxes.cls.cpu().numpy().astype(int)

                    last_dets = [
                        (*box, cls_id)
                        for box, cls_id in zip(boxes, clss)
                    ]

            # ===== 시각화 (로컬 디버깅용) =====
            if last_dets:
                for x1, y1, x2, y2, cls_id in last_dets:
                    label = names.get(cls_id, str(cls_id))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        frame, label, (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

            cv2.imshow("YOLO STREAM", frame)
            if cv2.waitKey(1) == 27:
                break

    finally:
        process.terminate()
        cv2.destroyAllWindows()
        print("🛑 FFmpeg YOLO 종료")
