# =========================================================
# Sleep Pose Recognition - YOLO Pose + CNN Hybrid (Robust Version)
# Author: ChatGPT
# =========================================================
# IMPORTANT:
# This file is designed to FAIL GRACEFULLY when PyTorch is not installed.
# If torch is missing, it will:
#   1) Print clear installation instructions
#   2) Exit without crashing
#
# Expected usage:
#   - Run in an environment where PyTorch + torchvision are installed
#   - This file is NOT meant to run inside minimal sandbox environments
# =========================================================
from ultralytics import YOLO
import os
import sys
import cv2
import numpy as np

# =========================================================
# Dependency Guard (CRITICAL FIX)
# =========================================================

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
except ModuleNotFoundError as e:
    print("\n[ERROR] Required dependency not found:")
    print(e)
    print("\nThis project requires PyTorch and torchvision.")
    print("Please install them in a proper environment, for example:\n")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("  # or CPU-only:")
    print("  pip install torch torchvision")
    print("\nAfter installation, re-run this script.\n")
    sys.exit(0)

# =========================================================
# Utils
# =========================================================

def load_yolo_pose_label(label_path, img_w, img_h):
    """
    Reads a YOLO-Pose label file and returns:
    - bbox (pixel coords)
    - normalized keypoints (17x3)
    - class id
    """

    data = np.loadtxt(label_path).reshape(-1)

    if data.shape[0] < 5 + 17 * 3:
        raise ValueError(f"Invalid label format: {label_path}")

    cls = int(data[0])
    xc, yc, w, h = data[1:5]

    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)




########################
    # # 1. 헤더 이후의 모든 데이터 가져오기 (키포인트 후보들)
    # raw_kpts = data[5:]
    # if raw_kpts.size!= 51:
    #     print("■■■",raw_kpts.size,label_path)
    # # 2. 우리에게 필요한 건 딱 51개 (17개 * 3값)
    # target_len = 17 * 3
    #
    # # 3. [핵심] 슬라이싱으로 처리 (길면 자르고, 짧으면 패딩)
    # if len(raw_kpts) >= target_len:
    #     # 107개든 1000개든 앞에서 51개만 뚝 자름 -> 성공
    #     real_kpts = raw_kpts[:target_len]
    #     # print("1■■■■",label_path)
    # else:
    #     # 51개보다 부족하면 뒤를 0으로 채움 (안전장치)
    #     pad_len = target_len - len(raw_kpts)
    #     real_kpts = np.pad(raw_kpts, (0, pad_len), constant_values=0)
    #     # print("2■■■■",label_path)
#################################






    kpts = data[5:].reshape(17, 3)

    kpts_norm = []
    for x, y, c in kpts:
        kx = (x - xc) / max(w, 1e-6)
        ky = (y - yc) / max(h, 1e-6)
        kpts_norm.append([kx, ky, c])

    return (x1, y1, x2, y2), np.array(kpts_norm, dtype=np.float32), cls


def crop_image(img, bbox):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        # print(img_path)
        raise ValueError("Invalid crop region")

    return img[y1:y2, x1:x2]

# =========================================================
# Dataset
# =========================================================
class SleepPoseDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
        if not self.images:
            raise RuntimeError("No images found in dataset")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        # print(img_path)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        #라벨정보가져와서 이미지crop
        bbox, kpts, cls = load_yolo_pose_label(label_path, w, h)
        crop = crop_image(img, bbox)

        if self.transform:
            crop = self.transform(crop)

        return crop, torch.from_numpy(kpts), torch.tensor(cls, dtype=torch.long)

# =========================================================
# Model
# =========================================================

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)#tf_EfficientNet_21k huggingface(timm)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class KeypointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(17 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        self.out_dim = 256

    def forward(self, kpts):
        return self.net(kpts.view(kpts.size(0), -1))


class SleepPoseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.kpt_enc = KeypointEncoder()

        self.classifier = nn.Sequential(
            nn.Linear(self.img_enc.out_dim + self.kpt_enc.out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, kpts):
        f_img = self.img_enc(img)
        f_kpt = self.kpt_enc(kpts)
        return self.classifier(torch.cat([f_img, f_kpt], dim=1))

# =========================================================
# Training
# =========================================================

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # label, image읽어옴, img crop
    dataset = SleepPoseDataset('..\\img\\pose\\train\\images', '..\\img\\pose\\train\\labels', transform)
    val_dataset = SleepPoseDataset('..\\img\\pose\\val\\images', '..\\img\\pose\\val\\labels', transform)

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=12)
    # efficientnet_v2_s
    model = SleepPoseNet(num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    class_weights = torch.tensor([
        1.0,  # lying
        1.2,  # side
        1.5,  # handup
        1.5,  # back
        0.6  # other
    ], device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(15):
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
        val_loss, class_counts = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Pred Dist: {class_counts}"
        )

    torch.save(model.state_dict(), 'sleep_pose_hybrid2.pt')
 # ---------------------------------------------------------
# Basic Inference Input Builder (IMAGE + YOLO OUTPUT)
# ---------------------------------------------------------

def build_hybrid_inputs(image_bgr, bbox, kpts, device):
    """
    Prepare inputs for SleepPoseNet prediction.

    Args:
        image_bgr (np.ndarray): original BGR image
        bbox (tuple): (x1, y1, x2, y2) from YOLO
        kpts (np.ndarray): (17,3) YOLO keypoints (normalized)
        device: torch device

    Returns:
        img_tensor: (1,3,224,224)
        kpt_tensor: (1,17,3)
    """
    # Crop person region
    crop = crop_image(image_bgr, bbox)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(crop).unsqueeze(0).to(device)
    kpt_tensor = torch.from_numpy(kpts).unsqueeze(0).to(device)

    return img_tensor, kpt_tensor


# ---------------------------------------------------------
# Final Prediction with Distinction
# ---------------------------------------------------------

def predict_with_distinction(model, img, kpts, device, conf_thres=0.5):
    """
    Perform prediction with post-hoc distinction rules.

    Distinction strategy:
    1) Base decision from neural network logits
    2) Rule-based refinement using keypoints geometry

    Returns:
        class_id (int)
    """
    model.eval()
    with torch.no_grad():
        logits = model(img.to(device), kpts.to(device))
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs))

    # -------------------------------
    # Rule-based distinction (CRITICAL)
    # -------------------------------
    kp = kpts[0].cpu().numpy()  # (17,3)

    # Key joints (COCO)
    nose = kp[0]
    l_shoulder, r_shoulder = kp[5], kp[6]
    l_hip, r_hip = kp[11], kp[12]
    l_wrist, r_wrist = kp[9], kp[10]

    shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
    torso_height = abs((l_shoulder[1] + r_shoulder[1]) / 2 - (l_hip[1] + r_hip[1]) / 2)

    # Hand-up override
    hand_up = (
        l_wrist[1] < nose[1] or r_wrist[1] < nose[1]
    )

    # Side vs Lying distinction
    side_pose = shoulder_width < torso_height * 0.6

    # Back (prone) heuristic: head lower than hips
    back_pose = nose[1] > (l_hip[1] + r_hip[1]) / 2

    # Apply overrides only when confidence is weak
    if probs[pred] < conf_thres:
        if hand_up:
            return 2  # handup
        if back_pose:
            return 3  # back
        if side_pose:
            return 1  # side
        return 0  # lying

    return pred



# =========================================================
# Inference Example (YOLO-Pose → Hybrid Prediction)
# =========================================================

def predict(image_path, yolo_weights="best_hj.pt", hybrid_weights="sleep_pose_hybrid2.pt"):
    """
    End-to-end inference:
    - Read image
    - Run YOLO-Pose
    - Build hybrid inputs
    - Predict sleep pose with distinction
    """
    from ultralytics import YOLO

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -----------------------------
    # 1) Load Hybrid Model
    # -----------------------------
    hybrid_model = SleepPoseNet(num_classes=5).to(device)
    hybrid_model.load_state_dict(torch.load(hybrid_weights, map_location=device))

    # -----------------------------
    # 2) Load YOLO-Pose Model
    # -----------------------------
    yolo_model = YOLO(yolo_weights)

    # -----------------------------
    # 3) Read Image
    # -----------------------------
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # -----------------------------
    # 4) Run YOLO-Pose
    # -----------------------------
    results = yolo_model(image, conf=0.5, iou=0.5)
    result = results[0]

    if len(result.boxes) == 0:
        print("[INFO] No person detected.")
        return None

    # Use the highest confidence person
    bbox_xyxy = result.boxes.xyxy[0].cpu().numpy().astype(int)
    bbox = tuple(bbox_xyxy)

    if result.keypoints is None:
        print("[INFO] No keypoints detected.")
        return None

    # YOLO keypoints: (17,2) → expand to (17,3)
    kpts_xy = result.keypoints.xy[0].cpu().numpy()
    # conf = np.ones((17, 1), dtype=np.float32)
    # kpts = np.concatenate([kpts_xy, conf], axis=1).astype(np.float32)
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)

    kpts_norm = []
    for x, y in kpts_xy:
        kx = (x - (x1 + x2) / 2) / bw
        ky = (y - (y1 + y2) / 2) / bh
        kpts_norm.append([kx, ky, 1.0])

    kpts = np.array(kpts_norm, dtype=np.float32)

    # -----------------------------
    # 5) Build Hybrid Inputs
    # -----------------------------
    img_t, kpt_t = build_hybrid_inputs(image, bbox, kpts, device)

    # -----------------------------
    # 6) Predict with Distinction
    # -----------------------------
    cls_id = predict_with_distinction(hybrid_model, img_t, kpt_t, device)

    return cls_id

def predictERror():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Load trained hybrid model
    model = SleepPoseNet(num_classes=5).to(device)
    model.load_state_dict(torch.load("sleep_pose_hybrid.pt", map_location=device))
    images = "..\\img\\pose\\test\\images\\KakaoTalk_20240328_004945964_10-fotor-2024032812015_jpg.rf.9b5cb22ddf1092b92f2d2fa65f76ceb5.jpg"
    # 2) YOLO-Pose 결과 (예시)
    image = cv2.imread(images)  # BGR
    model = YOLO('best.pt')
    results = model(image, conf=0.5, iou=0.5)
    result = results[0]
    if len(result.boxes) > 0:
        # 3. 가장 확률(conf)이 높은 첫 번째 사람의 박스 가져오기
        bbox = result.boxes[0]
        x1, y1, x2, y2 = bbox.xyxy[0].cpu().numpy().astype(int)

        print(f"찾은 좌표: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        # 2. 키포인트가 있는지 확인
        if result.keypoints is not None:
            # .xy : (x, y) 좌표만 꺼냄 (Shape: [사람수, 17, 2])
            # .cpu().numpy() : 텐서를 넘파이로 변환
            kpts = result.keypoints.xy[0].cpu().numpy()  # 첫 번째 사람의 (17, 2)

            # 3. 모델에 넣기 위해 1줄로 펴기 (34개)
            kpts_flat = kpts.flatten()  # (34,)

            print(f"추출 성공! 모양: {kpts.shape} -> {kpts_flat.shape}")

        else:
            # 키포인트 못 찾았으면 0으로 채움
            kpts_flat = np.zeros(34, dtype=np.float32)
            print("키포인트 없음. 0으로 채움.")
    else:
        print("사람을 못 찾았습니다.")

    # 3) Build inputs
    img_t, kpt_t = build_hybrid_inputs(image, bbox, kpts, device)

    # 4) Predict
    cls_id = predict_with_distinction(model, img_t, kpt_t, device)

    print("Predicted class:", cls_id)



def validate(model, dataloader, criterion, device):
    """
    Validation loop with class-wise statistics.

    Returns:
        avg_loss (float)
        class_counts (dict)
    """
    model.eval()
    total_loss = 0.0
    preds_all = []

    with torch.no_grad():
        for imgs, kpts, labels in dataloader:
            imgs = imgs.to(device)
            kpts = kpts.to(device)
            labels = labels.to(device)

            logits = model(imgs, kpts)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            preds_all.append(preds.cpu())

    preds_all = torch.cat(preds_all)
    unique, counts = torch.unique(preds_all, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))

    avg_loss = total_loss / len(dataloader)
    return avg_loss, class_counts

# =========================================================
# Minimal Self-Test (NO TRAINING)
# =========================================================

def _self_test():
    """Lightweight sanity check (no dataset required)."""
    model = SleepPoseNet(num_classes=5)
    img = torch.randn(1, 3, 224, 224)
    kpt = torch.randn(1, 17, 3)
    out = model(img, kpt)
    assert out.shape == (1, 5)
    print("[OK] Model forward pass works.")

def predict_video(
    video_path,
    yolo_weights="best_hj.pt",
    hybrid_weights="sleep_pose_hybrid.pt",
    output_path=None,
    conf_thres=0.5,
    stream=True
):
    """
    Predict sleep pose for each frame in a video and STREAM results.

    - Draws bbox
    - Draws keypoints & skeleton
    - Shows window in real-time if stream=True
    """
    from ultralytics import YOLO

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hybrid_model = SleepPoseNet(num_classes=5).to(device)
    hybrid_model.load_state_dict(torch.load(hybrid_weights, map_location=device))
    hybrid_model.eval()

    yolo_model = YOLO(yolo_weights)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    writer = None
    if output_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

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

        cls_id = None
        results = yolo_model(frame, conf=0.5, iou=0.5, verbose=False)
        result = results[0]

        if len(result.boxes) > 0 and result.keypoints is not None:
            bbox_xyxy = result.boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox_xyxy
            bbox = tuple(bbox_xyxy)

            kpts_xy = result.keypoints.xy[0].cpu().numpy()
            conf = np.ones((17, 1), dtype=np.float32)
            kpts = np.concatenate([kpts_xy, conf], axis=1).astype(np.float32)

            img_t, kpt_t = build_hybrid_inputs(frame, bbox, kpts, device)
            frame_count += 1
            if frame_count % 10 != 0:  # 3프레임중 1프레임만처리
                # print(frame)
                continue

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
            for i, (x, y, c) in enumerate(kpts):
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




if __name__ == '__main__':
    # if os.environ.get('SLEEPPOSE_SELFTEST') == '1':
    #     _self_test()
    # else:
    #     train()

    #predict image
    # cls_id = predict(
    #     image_path="..\\img\\pose\\test\\images\\RUIDc20a26f9d8e04402a6aeca205e669e73_jpg.rf.284ee678a5da2a353f6f21f1e17bb803.jpg",
    #     yolo_weights="best_hj.pt",
    #     hybrid_weights="sleep_pose_hybrid2.pt"
    # )
    # print("Predicted class:", cls_id)

    #predict video
    preds = predict_video(
        video_path="D:\\project\\github\\Dogwalk\\vision\\img\\video\\KakaoTalk02.mp4",
        yolo_weights="best_hj.pt",
        hybrid_weights="sleep_pose_hybrid2.pt",
        output_path="D:\\project\\github\\Dogwalk\\vision\\img\\video\\KakaoTalk02_output.mp4"
    )
  # 0: lying    # 누움
  # 1: side     # 옆모습
  # 2: handup   # 손듦
  # 3: back     # 뒷모습
  # 4: other    # 기타
