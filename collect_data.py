import cv2, csv, os
import mediapipe as mp
from datetime import datetime

print("=== 낙상 데이터 수집 라벨 선택 ===")
print("0 - 정상")
print("1 - 낙상 (머리)")
print("2 - 낙상 (손목)")
print("3 - 낙상 (골반)")
print("4 - 낙상 (기타)")
choice = input("수집할 데이터 유3" \
"형 번호를 입력하세요: ")

if choice == "0":
    LABEL, PART = 0, 3
elif choice == "1":
    LABEL, PART = 1, 0
elif choice == "2":
    LABEL, PART = 1, 1
elif choice == "3":
    LABEL, PART = 1, 2
elif choice == "4":
    LABEL, PART = 1, 3
else:
    print("❌ 잘못된 입력입니다. 종료합니다.")
    exit()

print(f"✅ 선택된 라벨: {'정상' if LABEL == 0 else '낙상'} / 부위: {PART}")

SAVE_DIR = "pose_data"
SEQUENCE_LENGTH = 30
SELECTED_IDX = [0, 10, 15, 16, 23, 24]

os.makedirs(SAVE_DIR, exist_ok=True)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sequence = []
prev_zs = None

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        # ✅ 랜드마크 시각화
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        keypoints = []
        current_zs = []

        for idx in SELECTED_IDX:
            lm = result.pose_landmarks.landmark[idx]
            current_zs.append(lm.z)

        for i, idx in enumerate(SELECTED_IDX):
            lm = result.pose_landmarks.landmark[idx]
            z_now = lm.z
            z_prev = prev_zs[i] if prev_zs else z_now
            speed_z = z_now - z_prev
            keypoints.extend([lm.x, lm.y, lm.z, speed_z])
        prev_zs = current_zs
        sequence.append(keypoints)

        if len(sequence) == SEQUENCE_LENGTH:
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(os.path.join(SAVE_DIR, filename), "w", newline='') as f:
                writer = csv.writer(f)
                for row in sequence:
                    writer.writerow(row + [LABEL, PART])
            print(f"✅ 저장 완료: {filename}")
            sequence = []

    cv2.imshow("🟢 데이터 수집 중 (미디어파이프 시각화)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
