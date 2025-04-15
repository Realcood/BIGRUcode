import cv2
import torch
import numpy as np
import mediapipe as mp
from model import FallBiGRUAttentionNet
from PIL import ImageFont, ImageDraw, Image
import os

# ✅ 사용할 관절 index
SELECTED_IDX = [0, 10, 15, 16, 23, 24]
PART_MAP = {0: "머리", 1: "손목", 2: "골반", 3: "기타"}

# ✅ 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    font = ImageFont.truetype(font_path, 32)
else:
    print("⚠️ 한글 폰트가 없어 기본 폰트로 출력됩니다.")
    font = ImageFont.load_default()

# ✅ 모델 로드
model = FallBiGRUAttentionNet(input_dim=24, hidden_dim=128, num_layers=2)
model.load_state_dict(torch.load("fall_bigru_model.pth", map_location="cpu"))
model.eval()

# ✅ MediaPipe 초기화
pose = mp.solutions.pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# ✅ 종료 신호 변수
exit_flag = [False]  # 리스트로 감싸서 내부 함수에서도 수정 가능

# ✅ 마우스 콜백 함수 정의
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:  # 마우스 우클릭
        print("🖱️ 마우스 우클릭으로 종료합니다.")
        exit_flag[0] = True

# ✅ 창 초기화 및 마우스 콜백 등록
cv2.namedWindow("🟢 실시간 낙상 감지 (한글 표시 OK)")
cv2.setMouseCallback("🟢 실시간 낙상 감지 (한글 표시 OK)", mouse_callback)

# ✅ 카메라 시작
cap = cv2.VideoCapture(0)
sequence = []
prev_zs = None
last_label = "분석 중..."
last_part = "-"

print("📸 실시간 낙상 감지를 시작합니다. (q 또는 우클릭으로 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 카메라 프레임을 불러올 수 없습니다.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        keypoints = []
        current_zs = []

        for idx in SELECTED_IDX:
            lm = result.pose_landmarks.landmark[idx]
            current_zs.append(lm.z)

        for i, idx in enumerate(SELECTED_IDX):
            lm = result.pose_landmarks.landmark[idx]
            z_now = lm.z
            z_prev = prev_zs[i] if prev_zs else z_now
            z_speed = z_now - z_prev
            keypoints.extend([lm.x, lm.y, lm.z, z_speed])
        prev_zs = current_zs

        sequence.append(keypoints)
        if len(sequence) > 30:
            sequence.pop(0)

        if len(sequence) == 30:
            input_tensor = torch.tensor([sequence], dtype=torch.float32)
            with torch.no_grad():
                fall_out, part_out = model(input_tensor)
                fall_pred = torch.argmax(fall_out, 1).item()
                part_pred = torch.argmax(part_out, 1).item()

            last_label = "💥 낙상 발생!" if fall_pred == 1 else "정상입니다"
            last_part = PART_MAP[part_pred]

    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)
    text = f"낙상 발생 : {last_part}" if last_label.startswith("💥") else "정상입니다"
    color = (0, 0, 255) if last_label.startswith("💥") else (0, 255, 0)

    text_size = draw.textbbox((0, 0), text, font=font)
    box_w = text_size[2] - text_size[0]
    box_h = text_size[3] - text_size[1]
    draw.rectangle([(25, 25), (25 + box_w + 10, 25 + box_h + 10)], fill=(0, 0, 0, 180))
    draw.text((30, 30), text, font=font, fill=color)

    frame = np.array(frame_pil)
    cv2.imshow("🟢 실시간 낙상 감지 (한글 표시 OK)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or exit_flag[0]:
        print("🛑 종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()
