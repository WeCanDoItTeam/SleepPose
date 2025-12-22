import subprocess
import numpy as np
import cv2
import datetime
from ultralytics import YOLO
import torch
from torchvision import transforms
import torch.nn as nn
import timm


# 디버그 모드 (비디오 재생)
DEBUG_MODE = True


# 설정값
WIDTH, HEIGHT = 640, 640
FRAME_SKIP = 5  # 15fps 중 3fps만 처리하기 위해 5프레임당 1회 추론
FRAME_SIZE = WIDTH * HEIGHT * 3
OFFSET = 9      # 약 3초(3fps * 3s) 동안 자세가 유지되어야 변경으로 인정
INF = -123456789
CONF_THRES = 0.5
IOU_THRES = 0.5


# 추론용: 크롭된 이미지 + (바운딩 박스 + 키포인트)
def build_hybrid_inputs(image_bgr, bbox, bbox_n, kpts_tensor, device):
    # Crop person region
    crop = crop_image(image_bgr, bbox)
    if crop is None : return None, None
    img_tensor = crop.unsqueeze(0).to(device)

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
    
    # if probs[pred] < CONF_THRES:
    #     pred = rule_based_postprocess(kpts)
        
    return pred # 신뢰도가 높으면 그대로 반환

# 룰 기반 결과 보정
def rule_based_postprocess(kpts_tensor):

    kpts = kpts_tensor.detach().cpu().numpy().flatten()

    if kpts.size != 55:
        return 4
     
    # kpts 55개
    kpts = kpts[4:]
    kpts = kpts.reshape(17,3)

    nose = kpts[0]
    l_shoulder, r_shoulder = kpts[5], kpts[6]
    l_wrist, r_wrist = kpts[9], kpts[10]
    l_hip, r_hip = kpts[11], kpts[12]

    shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
    torso_vec = np.array([(l_hip[0]+r_hip[0])/2 - (l_shoulder[0]+r_shoulder[0])/2,
                          (l_hip[1]+r_hip[1])/2 - (l_shoulder[1]+r_shoulder[1])/2])
    torso_len = np.linalg.norm(torso_vec)
    torso_angle = np.arctan2(torso_vec[1], torso_vec[0]) * 180 / np.pi

    # -----------------------------
    # 1) Lying 먼저 체크 (바로 누움)
    # torso 거의 수평, shoulder 넓음, 손목이 얼굴 위가 아닌 경우
    if abs(torso_angle) < 20 and shoulder_width > torso_len * 0.5 and \
       not ((l_wrist[2] > CONF_THRES and l_wrist[1] < l_shoulder[1]) or \
            (r_wrist[2] > CONF_THRES and r_wrist[1] < r_shoulder[1])):
        return 0  # lying

    # -----------------------------
    # 2) Hand-up
    if (l_wrist[2] > CONF_THRES and l_wrist[1] < l_shoulder[1]) or \
       (r_wrist[2] > CONF_THRES and r_wrist[1] < r_shoulder[1]):
        return 2  # handup

    # -----------------------------
    # 3) Back (엎드림)
    if nose[2] < CONF_THRES and (l_shoulder[2] > CONF_THRES or r_shoulder[2] > CONF_THRES):
        return 3  # back

    # -----------------------------
    # 4) Side
    if 45 < abs(torso_angle) < 135 and shoulder_width < torso_len * 0.7:
        return 1  # side

    # -----------------------------
    # 5) 기타
    return 4

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

# CNN
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=True)
        self.out_dim = 1280
        
        # 가중치 동결
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model.forward_features(x) # (Batch, 1280, 7, 7) 형태
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        return x.flatten(1)

# MLP
class KeypointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + 17 * 3, 128),
            nn.BatchNorm1d(128), # 학습 안정성을 위해 추가 권장
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.out_dim = 256

    def forward(self, kpts):
        return self.net(kpts.flatten(1))

# 모델 본체
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



device = 'cuda' if torch.cuda.is_available() else 'cpu'
hybrid_weights = r"C:\Users\USER\Documents\Github\SleepPose\Inference_Server\pose_pt\pose_4_22e_rl1e-4_best\sleep_pose_best_model.pt"

# ===== 추론 모델 로드 =====
hybrid_model = SleepPoseNet(num_classes=5).to(device)
hybrid_model.load_state_dict(torch.load(hybrid_weights, map_location=device))
hybrid_model.eval()

# ===== YOLO 모델 로드 =====
yolo_model = YOLO("yolo11n-pose.pt")

def save_to_mariadb(user_id, sleep_data_list):
    """
    마리아디비 저장 메서드 (내용은 나중에 채움)
    sleep_data_list: [{'pose': '자세명', 'start': '시간', 'end': '시간'}, ...]
    """
    print(f"\n💾 [DB 저장] 유저 {user_id}의 수면 기록 {len(sleep_data_list)}건 저장 시도 중...")
    # SQL 연결 및 INSERT 로직이 들어갈 자리
    for data in sleep_data_list:
        print(f" > {data['pose']}: {data['start']} ~ {data['end']}")

def run_ffmpeg_yolo(rtsp_url: str, ffmpeg_path: str, stop_flag: callable, user_id: int):

    if DEBUG_MODE:
        cap = cv2.VideoCapture(r"C:\Users\USER\Documents\Github\SleepPose\Inference_Server\data\lee_video\infer_Oh.mp4")
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
    
    current_pose = INF
    start_time = datetime.datetime.now()
    
    pending_pose = None  # 새로 바뀐 것처럼 보이는 자세
    pending_start_time = None
    consistent_count = 0  # 해당 자세가 몇 번 지속되었는지 카운트

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
        while not stop_flag():
            if DEBUG_MODE:
                ret, frame = cap.read()
                if not ret or frame is None: 
                    print("🛑 DEBUG video ended") 
                    break
            else:
                raw_frame = process.stdout.read(FRAME_SIZE)
                if len(raw_frame) != FRAME_SIZE: break
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()

            frame_count += 1
            # 15fps 중 3fps 추론 (5프레임마다 1번)
            if frame_count % FRAME_SKIP != 0:
                continue

            now = datetime.datetime.now()

            # ===== YOLO 추론 =====
            results = yolo_model(frame, imgsz=640, device=0, half=True, verbose=False, conf=CONF_THRES, iou=IOU_THRES)
            result = results[0]

            # 1. 자세 결정 (사람 유무에 따라)
            if len(result.boxes) > 0 and result.keypoints is not None:
                # [사람이 있을 때] 기존 GPU 최적화 로직 그대로 수행
                bbox_xyxy = result.boxes.xyxy[0]
                x1, y1, x2, y2 = bbox_xyxy.int().tolist()
                bbox_pixel = (x1, y1, x2, y2)
                bbox_norm = result.boxes.xyxyn[0]
                kpts_norm = result.keypoints.xyn[0]
                kpts_conf = result.keypoints.conf[0].unsqueeze(1)
                kpts_n = torch.cat([kpts_norm, kpts_conf], dim=1)

                img_t, kpt_t = build_hybrid_inputs(frame, bbox_pixel, bbox_norm, kpts_n, device)
                if img_t is None or kpt_t is None: continue
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
                
                if consistent_count >= OFFSET:
                    if current_pose != INF:
                        sleep_timeline.append({
                            'pose': current_pose,
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

    finally:
        
        # 반복문 종료 시 마지막 자세 저장
        if current_pose != INF:
            sleep_timeline.append({
                'pose': current_pose,
                'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        if stop_flag() and not DEBUG_MODE:
            process.terminate()
        cv2.destroyAllWindows() # (디버깅용)
        
        # 차곡차곡 쌓인 데이터를 DB로 전송
        if sleep_timeline:
            save_to_mariadb(user_id, sleep_timeline)
        
        print("🛑 분석 프로세스 종료")


