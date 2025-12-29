import subprocess
import numpy as np
import cv2
from datetime import datetime
from ultralytics import YOLO
import torch
from torchvision import transforms
import torch.nn as nn
import timm
from Inference_Server.inference.db_utils import get_db_connection

# 디버그 모드 (True : 비디오 재생 / False : RTSP)
DEBUG_MODE = True

# 설정값
WIDTH, HEIGHT = 640, 640
FRAME_SKIP = 5  # 총 15fps 중 3fps만 처리하기 위해 5프레임당 1회 추론
FRAME_SIZE = WIDTH * HEIGHT * 3
INF = -123456789 # Pose_id 초기값
CONF_THRES = 0.7 # 키포인트 신뢰도 기준
IOU_THRES = 0.5 # yolo용 iou 기준
KPT_ALPHA = 0.85 # 키포인트 스무스 이동을 위한 조정값
SAVE_INTERVAL_SEC = 60 * 5  # 5분마다 중간 저장

# =========================================================
# Utils
# =========================================================

# 추론용: 크롭된 이미지 + (바운딩 박스 + 키포인트)
def build_hybrid_inputs(image_bgr, bbox, bbox_n, kpts_tensor, device):
    # 바운딩박스 기준 이미지 크롭 (사람만 보이게)
    crop = crop_image(image_bgr, bbox)
    if crop is None : return None, None
    img_tensor = crop.unsqueeze(0).to(device)

    # 바운딩 박스 + 키포인트(정규화 됨)
    kpts_flat = kpts_tensor.reshape(-1) # (51,)
    kpts_add = torch.cat([bbox_n, kpts_flat], dim=0) # 바운딩 박스 추가
    kpt_tensor = kpts_add.unsqueeze(0).float() # (1, 55)

    return img_tensor, kpt_tensor

# 예측 결과 키포인트 처리
def predict_with_distinction(model, img, kpts, device):
    model.eval()
    with torch.no_grad():
        logits = model(img.to(device), kpts.to(device))
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs))
    
    # 일정 미만 신뢰도일 시 룰 기반 결과 보정 처리
    if probs[pred] < CONF_THRES:
        pred = rule_based_postprocess(kpts)
        
    return pred # 신뢰도가 높으면 그대로 반환

# 룰 기반 결과 보정
def rule_based_postprocess(kpts, conf_thres=0.4, shoulder_parallel_deg=20):
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.detach().cpu()

    # batch 차원 제거
    if kpts.ndim == 2:
        kpts = kpts[0]

    if kpts.numel() != 55:
        print(f"[DEBUG] invalid kpts numel = {kpts.numel()}")
        return 4

    kpts = kpts.numpy()

    # bbox 제거
    kpts = kpts[4:].reshape(17, 3)

    # 주요 키포인트
    nose = kpts[0]
    l_eye, r_eye = kpts[1], kpts[2]
    l_shoulder, r_shoulder = kpts[5], kpts[6]
    l_wrist, r_wrist = kpts[9], kpts[10]

    # =========================
    # STEP 1. 앞/뒤 판별
    # =========================
    face_conf_cnt = sum([
        nose[2] > conf_thres,
        l_eye[2] > conf_thres,
        r_eye[2] > conf_thres
    ])

    is_front = face_conf_cnt >= 2

    # =========================
    # STEP 2. 앞을 보고 있는 경우
    # =========================
    if is_front:
        # 어깨선 기울기
        dx = r_shoulder[0] - l_shoulder[0]
        dy = r_shoulder[1] - l_shoulder[1]
        shoulder_angle = np.degrees(np.arctan2(dy, dx))

        is_parallel = abs(shoulder_angle) < shoulder_parallel_deg

        # 손목 위치
        wrist_up = (
            (l_wrist[2] > conf_thres and l_wrist[1] < l_shoulder[1]) or
            (r_wrist[2] > conf_thres and r_wrist[1] < r_shoulder[1])
        )

        if is_parallel:
            return 2 if wrist_up else 0
        else:
            return 1  # 옆으로 누움

    # =========================
    # STEP 3. 얼굴 안 보임 → 엎드림
    # =========================
    return 3

# 이미지 크롭
def crop_image(img, bbox):
    if img is None:
        return None
    
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return transform(crop_rgb)

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
    if torch.isnan(kpts[5]).any() or torch.isnan(kpts[6]).any():
        return kpts
    # 양 어깨가 위치 뒤바뀌면 아예 키를 반대로 뒤집어 정상화 시킴
    if kpts[5,0] > kpts[6,0]:
        for a,b in pairs:
            kpts[[a,b]] = kpts[[b,a]]
    return kpts

# =========================================================
# Model
# =========================================================

# CNN
# 사용 모델 tf_efficientnetv2_s.in21k_ft_in1k
# 사전학습 imageNet 21k, 파인튜닝 imageNet 1k, 학습 시 - 2차 파인튜닝: top 레이어만 학습
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=True)
        self.out_dim = 1280
        
        # 가중치 동결
        for param in self.model.parameters():
            param.requires_grad = False

        # top 레이어만 학습
        for name, param in self.model.named_parameters():
            if "blocks.4" in name or "blocks.5" in name: # 끝부분만 동결 풀어서 학습시킴 (추론에선 eval 모드라 학습 안됨)
                param.requires_grad = True

    def forward(self, x):
        x = self.model.forward_features(x) # (Batch, 1280, 7, 7) 형태
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        return x.flatten(1)

# MLP
# (128) -> (256) -> (512)
class KeypointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + 17 * 3, 128),
            nn.BatchNorm1d(128), # 학습 안정성을 위해 추가 권장
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.out_dim = 512

    def forward(self, kpts):
        return self.net(kpts.flatten(1))

# 모델 본체
# CNN(1280) + MLP(512) -> 5
class SleepPoseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.kpt_enc = KeypointEncoder()

        self.classifier = nn.Sequential(
            nn.Linear(self.img_enc.out_dim + self.kpt_enc.out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, kpts):
        f_img = self.img_enc(img)
        f_kpt = self.kpt_enc(kpts)
        return self.classifier(torch.cat([f_img, f_kpt], dim=1))

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
hybrid_weights = r"./pose_pt/pose_9_22e_rl1e-4_best/sleep_pose_best_model.pt"

# ===== 추론 모델 로드 =====
hybrid_model = SleepPoseNet(num_classes=5).to(device)
hybrid_model.load_state_dict(torch.load(hybrid_weights, map_location=device))
hybrid_model.eval()

# ===== YOLO 모델 로드 =====
yolo_model = YOLO("yolo11n-pose.pt")


# =========================================================
# Main Method
# =========================================================

# 스레드 종료 시 정보 DB 저장
def save_to_mariadb(login_id, sleep_data_list):
    if not sleep_data_list:
        print("⚠️ 저장할 데이터가 없습니다.")
        return

    print(f"\n💾 [DB 저장] 유저 {login_id} 수면 기록 {len(sleep_data_list)}건 저장 시작")

    print("\n📋 [저장 예정 데이터 미리보기]")
    for i, data in enumerate(sleep_data_list, 1):
        print(
            f"{i:02d}. "
            f"user_id={login_id}, "
            f"pose={data['pose']}, "
            f"start={data['start']}, "
            f"end={data['end']}"
        )


    # 1️⃣ DB 연결
    conn = get_db_connection()

    if conn is None:
        print("❌ DB 연결 실패로 저장 중단")
        return

    try:
        with conn.cursor() as cur:
            insert_sql = """
            INSERT INTO sleep_pose (user_id, pose_class, st_dt, ed_dt, dt)
            VALUES (%s, %s, %s, %s, %s)
            """

            rows = []
            for data in sleep_data_list:
                rows.append((
                    login_id,
                    data['pose'],
                    datetime.fromisoformat(data['start']),
                    datetime.fromisoformat(data['end']),
                    datetime.now()
                ))

            # 2️⃣ 한 번에 INSERT
            cur.executemany(insert_sql, rows)
            conn.commit()

            print(f"✅ DB 저장 완료 ({len(rows)}건)")

    except Exception as e:
        conn.rollback()
        print("❌ DB 저장 실패:", e)

    finally:
        conn.close()

# [메인 함수] RTSP 실행 및 추론, 데이터 가공 저장
def run_ffmpeg_yolo(rtsp_url: str, ffmpeg_path: str, stop_flag: callable, login_id: int):

    OFFSET = 18 if DEBUG_MODE else 9 # 디버그 모드면 6초, RTSP면 3초
    # DEBUG_MODE일 시 비디오 추론
    if DEBUG_MODE:
        cap = cv2.VideoCapture(r".\data\lee_video\infer_Lee.mp4")
    else:
        cmd = [
            ffmpeg_path, "-rtsp_transport", "tcp", "-fflags", "nobuffer",
            "-flags", "low_delay", "-i", rtsp_url,
            "-vf", f"scale={WIDTH}:{HEIGHT}", "-pix_fmt", "bgr24",
            "-f", "rawvideo", "-"
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    # --- 자세 기록용 변수 ---
    sleep_timeline = []  # 최종 DB로 보낼 리스트
    
    last_save_time = datetime.now() 

    current_pose = INF
    start_time = datetime.now()
    
    pending_pose = None  # 새로 바뀐 것처럼 보이는 자세
    pending_start_time = None
    consistent_count = 0  # 해당 자세가 몇 번 지속되었는지 카운트
    prev_kpts_norm = None
    frame_count = 0
    print("✅ FFmpeg YOLO 스트림 및 타임라인 분석 시작")

    # COCO skeleton connections (디버깅용)
    skeleton = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]

    detected_pose = 4 # 기본값 (others)

    try:
        # stop_flag()이 람다 함수가 bool값이 바뀌는 것을 감지, inference_running = False 되기 이전까지 반복문이 실행
        # DEBUG_MODE일 땐 영상이 종료되었을 때 자동 종료
        while not stop_flag():
            if DEBUG_MODE:
                ret, frame = cap.read()
                if not ret or frame is None: 
                    print("🛑 DEBUG video ended") 
                    break
            else:
                raw_frame = process.stdout.read(FRAME_SIZE)
                if not raw_frame or len(raw_frame) < FRAME_SIZE:
                    print("❌ RTSP 프레임 수신 실패")
                    break

                if process.poll() is not None:
                    print("❌ FFmpeg 프로세스 종료")
                    break
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()

            frame = ir_preprocess(frame) # 적외선 환경 처리

            frame_count += 1
            # 15fps 중 3fps 추론 (5프레임마다 1번)
            if frame_count % FRAME_SKIP != 0:
                continue

            now = datetime.now()

            # ===== YOLO 추론 =====
            results = yolo_model(frame, imgsz=640, device=device, half=use_cuda, verbose=False, conf=CONF_THRES, iou=IOU_THRES)
            result = results[0]

            # 1. 자세 결정 (사람 유무에 따라)
            if len(result.boxes) > 0 and result.keypoints is not None:
                bbox_xyxy = result.boxes.xyxy[0]
                x1, y1, x2, y2 = bbox_xyxy.int().tolist()
                bbox_pixel = (x1, y1, x2, y2) # 원본 픽셀 기준 바운딩박스 (이미지 크롭에 필요)
                bbox_norm = result.boxes.xyxyn[0] # 정규화된 바운딩박스 (MLP 데이터에 필요)

                kpts_norm = result.keypoints.xyn[0] # 정규화된 키포인트 (17, 2) (MLP 데이터에 필요)
                kpts_norm = enforce_lr_consistency(kpts_norm) # 급격한 뒤집힘 방지
                kpts_norm = ema(prev_kpts_norm, kpts_norm, KPT_ALPHA) # 키포인트 스무스 이동
                prev_kpts_norm = kpts_norm.clone()
                kpts_conf = result.keypoints.conf[0].unsqueeze(1) # 키포인트 신뢰도 (17, 1)
                kpts_n = torch.cat([kpts_norm, kpts_conf], dim=1) # 정규화된 키포인트 + 신뢰도 (17, 3)

                # 이미지 크롭, MLP용 데이터 생성
                img_t, kpt_t = build_hybrid_inputs(frame, bbox_pixel, bbox_norm, kpts_n, device)
                if img_t is None or kpt_t is None: continue
                # 우리가 만든 모델에 추론
                detected_pose = predict_with_distinction(hybrid_model, img_t, kpt_t, device)
            else:
                # [사람이 없을 때] 강제로 Others(4) 처리
                detected_pose = 4

            # 2. 자세 변화 로직 (Offset 검증) 
            if detected_pose != current_pose:
                if detected_pose == pending_pose:
                    consistent_count += 1
                else:
                    pending_pose = detected_pose
                    pending_start_time = now
                    consistent_count = 1
                
                # 지정한 OFFSET 이상 자세가 유지되어야 이전 자세의 데이터 기록
                if consistent_count >= OFFSET:
                    if current_pose != INF:
                        sleep_timeline.append({
                            'pose': str(current_pose),
                            'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'end': pending_start_time.strftime('%Y-%m-%d %H:%M:%S')
                        })
                    current_pose = pending_pose
                    start_time = pending_start_time
                    consistent_count = 0
                    pending_pose = None
            else:
                consistent_count = 0
                pending_pose = None

            # ===============================
            # ⏱️ 주기적 DB 저장
            # ===============================
            if (now - last_save_time).total_seconds() >= SAVE_INTERVAL_SEC:
                if current_pose != INF:
                    sleep_timeline.append({
                        'pose': str(current_pose),
                        'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'end': now.strftime('%Y-%m-%d %H:%M:%S')
                    })

                if sleep_timeline:
                    print(f"⏱️ 중간 저장: {len(sleep_timeline)}건")
                    try:
                        save_to_mariadb(login_id, sleep_timeline)
                        sleep_timeline.clear()
                        last_save_time = now
                    except Exception as e:
                        print("⚠️ 중간 저장 실패, 다음 주기에 재시도", e)

    finally:
        # 반복문 종료 시 마지막 자세 저장
        if current_pose != INF:
            sleep_timeline.append({
                'pose': str(current_pose),
                'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        if stop_flag() and not DEBUG_MODE:
            process.terminate()
        cv2.destroyAllWindows() # (디버깅용)
        
        # 차곡차곡 쌓인 데이터를 DB로 전송
        if sleep_timeline:
            save_to_mariadb(login_id, sleep_timeline)
        
        print("🛑 분석 프로세스 종료")


