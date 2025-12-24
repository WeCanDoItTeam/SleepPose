from ultralytics import YOLO
import os
import sys
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from pathlib import Path
from tabulate import tabulate

# =========================================================
# Utils
# =========================================================

KPT_ALPHA = 0.85
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

# 기존 라벨을 (바운딩 박스(원본 이미지 픽셀) / 바운딩 박스(원본 이미지 정규화) + 키포인트(이미 정규화) / 클래스 아이디)로 분류
def load_yolo_pose_label(label_path, img_w, img_h):

    data = np.loadtxt(label_path).reshape(-1)

    if data.shape[0] < 5 + 17 * 3:
        raise ValueError(f"Invalid label format: {label_path}")

    # Class & BBox (normalized)
    cls = int(data[0])
    xc, yc, w, h = data[1:5]

    # BBox (pixel coords)
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)

    # Keypoints (already normalized)
    kpts = data[5:5 + 17 * 3].reshape(17, 3).astype(np.float32)

    # 🔒 안전 처리: 좌표 클리핑
    kpts[:, 0] = np.clip(kpts[:, 0], 0.0, 1.0)
    kpts[:, 1] = np.clip(kpts[:, 1], 0.0, 1.0)

    # bbox(정규화 된 상태) + kps => MLP용 데이터
    pose_feature = np.concatenate([
    np.array([xc, yc, w, h], dtype=np.float32), # (4,)
    kpts.flatten()  # (51,)
    ])  # → (55,)

    return (x1, y1, x2, y2), pose_feature, cls

# 이미지 크롭
def crop_image(img, bbox):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        # print(img_path)
        raise ValueError("Invalid crop region")

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

# 크롭된 이미지 기준 키포인트 변경
def normalize_kpts_to_crop(kpts, bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1e-6)
    bh = max(y2 - y1, 1e-6)

    kpts_crop = []
    kpts_only = kpts[4:].reshape(17,3).astype(np.float32)

    for x, y, c in kpts_only:
        if c == 0:
            kpts_crop.append([0.0, 0.0, 0])
            continue

        # ✅ 원본 이미지 기준 픽셀 좌표
        px = x * img_w
        py = y * img_h

        # crop 기준 정규화
        cx = (px - x1) / bw
        cy = (py - y1) / bh

        kpts_crop.append([
            np.clip(cx, 0.0, 1.0),
            np.clip(cy, 0.0, 1.0),
            c
        ])

    kpts_crop = np.array(kpts_crop, dtype=np.float32).flatten()

    return np.concatenate([kpts[:4], kpts_crop])

# =========================================================
# Dataset
# =========================================================

# train에서만 사용
class SleepPoseDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir

        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
        if not self.images:
            raise RuntimeError("No images found in dataset")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:    
            img_name = self.images[idx]
            img_path = os.path.join(self.img_dir, img_name)
            # print(img_path)
            label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
            img = cv2.imread(img_path)
            if img is None:
                raise self.__getitem__((idx + 1) % len(self))

            h, w = img.shape[:2]

            # 라벨정보가져와서 이미지crop
            bbox, kpts, cls = load_yolo_pose_label(label_path, w, h)
            crop = crop_image(img, bbox)
            # 크롭된 이미지로 kpts 재조정
            kpts_norm = normalize_kpts_to_crop(kpts, bbox, w, h)

            return crop, torch.from_numpy(kpts_norm), torch.tensor(cls, dtype=torch.long)
        
        except Exception as e:
            # 에러 발생 시 다음 인덱스로 넘어가기
            print(f"Skipping {self.images[idx]} due to error: {e}")
            return self.__getitem__((idx + 1) % len(self))

# =========================================================
# Model
# =========================================================

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

        # top 레이어만 학습
        for name, param in self.model.named_parameters():
            if "blocks.4" in name or "blocks.5" in name:
                param.requires_grad = True

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
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.out_dim = 512

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

from torch.nn import functional as F

# 변경한 손실함수
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean() if self.reduction=='mean' else loss.sum()

# =========================================================
# Training
# =========================================================

EPOCH = 25
BATCH_SIZE = 32

# 학습
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # label, image읽어옴, img crop
    dataset = SleepPoseDataset(r'C:\Users\USER\Documents\Github\SleepPose\Inference_Server\data\pose_Lee\train\images', 
                               r'C:\Users\USER\Documents\Github\SleepPose\Inference_Server\data\pose_Lee\train\labels')
    val_dataset = SleepPoseDataset(r'C:\Users\USER\Documents\Github\SleepPose\Inference_Server\data\pose_Lee\val\images', 
                                   r'C:\Users\USER\Documents\Github\SleepPose\Inference_Server\data\pose_Lee\val\labels')

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SleepPoseNet(num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # class_weights = torch.tensor([
    #     1.0,  # lying
    #     1.0,  # side
    #     1.0,  # handup
    #     1.0,  # back
    #     1.0  # other
    # ], device=device)

    criterion = FocalLoss()
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 🔥 최적 모델 저장을 위한 변수 초기화
    best_val_loss = float('inf')  # 처음에는 아주 큰 값으로 설정
    save_path = 'sleep_pose_best_model.pt'

    train_losses = []
    val_losses = []
    val_accs = []
    best_cm = None

    best_metrics = {
    "val_loss": None,
    "val_acc": None,
    "precision": None,
    "recall": None,
    "f1": None
    }

    for epoch in range(EPOCH):
        model.train() # 학습 모드   
        total_loss = 0.0
        for imgs, kpts, labels in loader:
            imgs, kpts, labels = imgs.to(device), kpts.to(device), labels.to(device)

            logits = model(imgs, kpts)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            train_loss = total_loss / len(loader)

        # 🔴 VALIDATION (epoch 끝나고 딱 1번)
        val_loss, val_acc, val_precision, val_recall, val_f1, cm = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Best model 저장 + best metrics 보관
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_cm = cm

            best_metrics["val_loss"] = val_loss
            best_metrics["val_acc"] = val_acc
            best_metrics["precision"] = val_precision
            best_metrics["recall"] = val_recall
            best_metrics["f1"] = val_f1

            torch.save(model.state_dict(), save_path)
            print(f"✨ Best model saved (Val Loss: {val_loss:.4f})")

    torch.save(model.state_dict(), 'sleep_pose_hybrid2_hj.pt')

    print("\n================ Final Metrics (Best Model) ================\n")

    table = [
        ["Train Loss (Last)", f"{train_losses[-1]:.4f}"],
        ["Best Val Loss", f"{best_metrics['val_loss']:.4f}"],
        ["Best Val Accuracy", f"{best_metrics['val_acc']:.4f}"],
        ["Precision (Macro)", f"{best_metrics['precision']:.4f}"],
        ["Recall (Macro)", f"{best_metrics['recall']:.4f}"],
        ["Macro F1-score", f"{best_metrics['f1']:.4f}"]
    ]

    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

    # ============================
    # 📊 Loss & Accuracy Plot
    # ============================
    epochs = range(1, EPOCH + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curve.jpg", dpi=300)
    plt.close()

    # ============================
    # 📊 혼동 행렬 저장
    # ============================
    class_names = ["lying", "side", "handup", "back", "other"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        best_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Best Val Loss Model)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_best.jpg", dpi=300)
    plt.close()

# ---------------------------------------------------------
# Basic Inference Input Builder (IMAGE + YOLO OUTPUT)
# ---------------------------------------------------------

# 추론용: 크롭된 이미지 + (바운딩 박스 + 키포인트)
def build_hybrid_inputs(image_bgr, bbox, bbox_n, kpts, device):
    # Crop person region
    crop = crop_image(image_bgr, bbox)

    img_tensor = crop.unsqueeze(0).to(device)

    kpts_flat = kpts.flatten() 

    kpts_add = np.concatenate([bbox_n, kpts_flat]) # 바운딩 박스 추가
    kpt_tensor = torch.from_numpy(kpts_add).unsqueeze(0).to(device) 

    return img_tensor, kpt_tensor

# ---------------------------------------------------------
# Final Prediction with Distinction
# ---------------------------------------------------------

# 예측 결과 키포인트 처리
def predict_with_distinction(model, img, kpts, device, conf_thres=0.7):
    model.eval()
    with torch.no_grad():
        logits = model(img.to(device), kpts.to(device))
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs))
    
    if probs[pred] < conf_thres:
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


# 학습에서 쓰이는 평가 메서드
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def validate(model, val_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for imgs, kpts, labels in val_loader:
            imgs, kpts, labels = imgs.to(device), kpts.to(device), labels.to(device)
            logits = model(imgs, kpts)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return val_loss, val_acc, precision, recall, f1, cm




# =========================================================
# Inference Example (YOLO-Pose → Hybrid Prediction)
# =========================================================

# 이미지용 추론 메서드

def predict_images(
    image_folder,
    yolo_weights="yolo11n-pose.pt",
    hybrid_weights="sleep_pose_hybrid_hj.pt",
    output_folder=None,
    conf_thres=0.3,
    stream=True
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hybrid_model = SleepPoseNet(num_classes=5).to(device)
    hybrid_model.load_state_dict(torch.load(hybrid_weights, map_location=device))
    hybrid_model.eval()

    yolo_model = YOLO(yolo_weights)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    skeleton = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]

    predictions = []

    img_paths = sorted(Path(image_folder).glob("*.[jp][pn]g"))  # jpg, png
    for img_path in img_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Failed to read image: {img_path}")
            predictions.append(None)
            continue

        cls_id = None
        try:
            results = yolo_model(frame, conf=conf_thres, iou=0.5, verbose=False)
            result = results[0]

            # 사람이 검출되지 않은 경우
            if len(result.boxes) == 0 or result.keypoints is None:
                print(f"No person detected in {img_path.name}")
                predictions.append(None)
                if output_folder is not None:
                    out_path = os.path.join(output_folder, img_path.name)
                    cv2.imwrite(out_path, frame)
                continue

            # 사람 검출 시 기존 추론 로직
            bbox_xyxy = result.boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox_xyxy
            bbox = tuple(bbox_xyxy)
            bbox_norm = result.boxes.xyxyn[0].cpu().numpy()

            kpts = result.keypoints.xy[0].cpu().numpy()
            kpts_norm = result.keypoints.xyn[0].cpu().numpy()
            kpts_conf = result.keypoints.conf[0].cpu().numpy().reshape(17, 1)
            kpts_n = np.concatenate([kpts_norm, kpts_conf], axis=1).astype(np.float32)

            img_t, kpt_t = build_hybrid_inputs(frame, bbox, bbox_norm, kpts_n, device)

            cls_id = predict_with_distinction(hybrid_model, img_t, kpt_t, device, conf_thres)
            print(f"{img_path.name}: class={cls_id}")

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"class={cls_id}", (x1, max(0, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            restore_kpts_to_original(kpt_t, bbox, conf_thres)

            # Draw keypoints
            for i, (x, y) in enumerate(kpts):
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

            # Draw skeleton
            for a, b in skeleton:
                x1_, y1_ = int(kpts[a][0]), int(kpts[a][1])
                x2_, y2_ = int(kpts[b][0]), int(kpts[b][1])
                cv2.line(frame, (x1_, y1_), (x2_, y2_), (255, 0, 0), 2)

            predictions.append(cls_id)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            predictions.append(None)

        # 저장
        if output_folder is not None:
            out_path = os.path.join(output_folder, img_path.name)
            cv2.imwrite(out_path, frame)

        # 화면 출력
        if stream:
            cv2.imshow("SleepPose Prediction", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
    return predictions


# 안 쓰임
# def predictERror():

# =========================================================
# Minimal Self-Test (NO TRAINING)
# =========================================================

from ultralytics import YOLO

# 비디오 추론
def predict_video(
    video_path,
    yolo_weights="yolo11n-pose.pt",
    hybrid_weights="sleep_pose_hybrid_hj.pt",
    output_path=None,
    conf_thres=0.7,
    stream=True
):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hybrid_model = SleepPoseNet(num_classes=5).to(device)
    hybrid_model.load_state_dict(torch.load(hybrid_weights, map_location=device))
    hybrid_model.eval()

    yolo_model = YOLO(yolo_weights)

    IMG_SIZE = 640

    prev_kpts_norm = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    writer = None
    if output_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (IMG_SIZE, IMG_SIZE))

    predictions = []

    # COCO skeleton connections
    skeleton = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            # print(frame)
            continue

        frame = ir_preprocess(frame) # 적외선 환경 처리
        cls_id = None
        results = yolo_model(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
        result = results[0] # 무조건 첫 번째 박스만 검출

        if len(result.boxes) > 0 and result.keypoints is not None:
            bbox_xyxy = result.boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox_xyxy
            bbox = tuple(bbox_xyxy) # 픽셀단위 (정규화 아님)
            bbox_norm = result.boxes.xyxyn[0].cpu().numpy() # 픽셀단위 (정규화 됨)

            kpts = result.keypoints.xy[0].cpu().numpy() # 키포인트 (정규화 안 됨) cv 출력용

            kpts_norm = result.keypoints.xyn[0].cpu().numpy() # 키포인트 정규화
            kpts_norm = enforce_lr_consistency(kpts_norm) # 급격한 뒤집힘 방지
            kpts_norm = ema(prev_kpts_norm, kpts_norm, KPT_ALPHA) # 키포인트 스무스 이동
            prev_kpts_norm = kpts_norm.copy()
            kpts_conf = result.keypoints.conf[0].cpu().numpy().reshape(17, 1) # 키포인트 신뢰도
            kpts_n = np.concatenate([kpts_norm, kpts_conf], axis=1).astype(np.float32)

            img_t, kpt_t = build_hybrid_inputs(frame, bbox, bbox_norm, kpts_n, device)


            # 모델 추론
            cls_id = predict_with_distinction(
                hybrid_model, img_t, kpt_t, device, conf_thres
            )
            print(cls_id)
            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"class={cls_id}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Draw keypoints
            for i, (x, y) in enumerate(kpts):
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

            # Draw skeleton
            for a, b in skeleton:
                x1_, y1_ = int(kpts[a][0]), int(kpts[a][1])
                x2_, y2_ = int(kpts[b][0]), int(kpts[b][1])
                cv2.line(frame, (x1_, y1_), (x2_, y2_), (255, 0, 0), 2)

        predictions.append(cls_id)

        if writer is not None:
            writer.write(frame)

        if stream:
            cv2.imshow("SleepPose Prediction", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    return predictions

def restore_kpts_to_original(kpt_t, bbox, conf_thres=0.5):
    """
    kpt_t: (55,) torch or np  (bbox + 17 keypoints, normalized to crop)
    bbox: (x1, y1, x2, y2) in original pixel
    return: list of (x, y, conf) in original pixel or None
    """
    if isinstance(kpt_t, torch.Tensor):
        kpt_t = kpt_t.detach().cpu().numpy()

    kpt_t = kpt_t.reshape(-1)
    if kpt_t.size != 55:
        return []

    x1, y1, x2, y2 = bbox
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)

    # bbox 정보 제거 (앞 4개)
    kpts = kpt_t[4:].reshape(17, 3)

    kpts_orig = []
    for x_n, y_n, c in kpts:
        if c < conf_thres:
            kpts_orig.append(None)
        else:
            x = int(x1 + x_n * crop_w)
            y = int(y1 + y_n * crop_h)
            kpts_orig.append((x, y, c))

    return kpts_orig

# 쿠다 사용 가능한지 확인
def check_device():
    print(f"PyTorch 버전: {torch.__version__}")
    
    # CUDA 사용 가능 여부
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 사용 가능 여부: {cuda_available}")
    
    if cuda_available:
        # 현재 선택된 GPU 이름
        print(f"현재 GPU 장치: {torch.cuda.get_device_name(0)}")
        # GPU 개수
        print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
        
        # 실제 텐서를 생성해서 전송 테스트
        test_tensor = torch.zeros(1).to('cuda')
        print(f"테스트 텐서 위치: {test_tensor.device}")
    else:
        print("GPU를 찾을 수 없습니다. CPU로 학습을 진행합니다.")

import matplotlib.pyplot as plt
import numpy as np

# 프레임 타임그래프 보여주기
def visualize_preds(preds, save_path="preds_timeline.jpg"):
    # None 제거 또는 -1로 치환
    preds_clean = [-1 if p is None else p for p in preds]
    frames = np.arange(len(preds_clean))

    plt.figure(figsize=(15, 4))
    plt.plot(frames, preds_clean, marker='o', linestyle='-', alpha=0.7)

    plt.yticks(
        ticks=[-1, 0, 1, 2, 3, 4],
        labels=["None", "lying", "side", "handup", "back", "other"]
    )

    plt.xlabel("Frame Index")
    plt.ylabel("Predicted Class")
    plt.title("Sleep Pose Prediction Timeline")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📊 Saved prediction timeline: {save_path}")

from collections import Counter

# 프레임 횟수 보여주기
def visualize_pred_distribution(preds, save_path="preds_distribution.jpg"):
    preds_clean = [p for p in preds if p is not None]
    counter = Counter(preds_clean)

    labels_map = {
        0: "lying",
        1: "side",
        2: "handup",
        3: "back",
        4: "other"
    }

    labels = [labels_map[k] for k in counter.keys()]
    values = list(counter.values())

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("Sleep Pose Prediction Distribution")
    plt.ylabel("Frame Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📊 Saved prediction distribution: {save_path}")

# 비율 계산해서 보여주기
def visualize_pred_ratio(preds, fps=30, save_path="preds_ratio.jpg"):
    preds_clean = [p for p in preds if p is not None]
    counter = Counter(preds_clean)

    labels_map = {
        0: "lying",
        1: "side",
        2: "handup",
        3: "back",
        4: "other"
    }

    labels = []
    times = []

    for k, v in counter.items():
        labels.append(labels_map[k])
        times.append(v / fps / 60)  # 분 단위

    plt.figure(figsize=(6, 6))
    plt.pie(times, labels=labels, autopct="%.1f%%", startangle=90)
    plt.title("Sleep Pose Time Ratio (minutes)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"🛌 Saved sleep pose ratio chart: {save_path}")

if __name__ == '__main__':

    FRAME_SKIP = 1
    CONF_THRES = 0.5
    IOU_THRES = 0.5

    name = "TEST_0.mp4"
    pt_name = "sleep_pose_best_model.pt"

    # 학습시키기
    # train()

    # 실행
    # check_device()

    # preds = predict_images(
    #     image_folder=r"C:\Users\USER\Documents\Github\SleepPose\Inference_Server\data\pose_Lee\test\images",
    #     yolo_weights="yolo11n-pose.pt",
    #     hybrid_weights=rf"C:\Users\USER\Documents\Github\SleepPose\Inference_Server\pose_pt\pose_3_18e_rl1e-4_best\{pt_name}",
    #     output_folder=rf"C:\Users\USER\Documents\Github\SleepPose\Inference_Server\infer_images"
    # )
    # # 클래스 ID → 클래스 이름 매핑
    # class_names = {
    #     0: "정자세",
    #     1: "옆으로 누운 자세",
    #     2: "손을 든 자세",
    #     3: "엎드린 자세",
    #     4: "그 외 자세"
    # }

    # # predictions: 이미지별 추론 결과 리스트
    # # img_paths: 이미지 파일 리스트 (같은 순서로 정렬되어 있어야 함)
    # img_paths = r"C:\Users\USER\Documents\Github\SleepPose\Inference_Server\data\pose_Lee\test\images"
    # for img_path, cls_id in zip(img_paths, preds):
    #     img_name = img_path.name
    #     if cls_id is None:
    #         print(f"{img_name}: 사람 없음")
    #     else:
    #         cls_name = class_names.get(cls_id, "알 수 없음")
    #         print(f"{img_name}: {cls_name} (클래스 ID: {cls_id})")


    # predict video
    preds = predict_video(
        video_path=rf"C:\Users\USER\Documents\Github\SleepPose\Inference_Server\data\lee_video\infer_Oh.mp4",
        yolo_weights="yolo11n-pose.pt",
        hybrid_weights=rf"C:\Users\USER\Documents\Github\SleepPose\Inference_Server\pose_pt\pose_9_22e_rl1e-4_best\{pt_name}",
        #output_path=rf"C:\Users\USER\Documents\Github\SleepPose\Inference_Server\infer_video\{name}"
    )
    visualize_preds(preds, save_path="sleep_pose_timeline.jpg")
    visualize_pred_distribution(preds)
    visualize_pred_ratio(preds, fps=30)