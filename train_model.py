import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from model import FallBiGRUAttentionNet

# ✅ 데이터셋 정의
class FallDataset(Dataset):
    def __init__(self, folder):
        self.data = []
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                values = pd.read_csv(os.path.join(folder, file), header=None).values
                input_seq = values[:, :-2]
                fall_label = int(values[0, -2])
                part_label = int(values[0, -1])
                self.data.append((input_seq, fall_label, part_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y1, y2 = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), y1, y2

# ✅ 데이터 로딩
train_loader = DataLoader(FallDataset("pose_data"), batch_size=32, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 및 손실 함수, 옵티마이저 정의
model = FallBiGRUAttentionNet(input_dim=24, hidden_dim=128, num_layers=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ✅ EarlyStopping 관련 변수 정의
best_loss = float('inf')
patience = 10  # 개선이 없을 때 몇 번 기다릴지
trigger = 0

# ✅ 학습 루프
for epoch in range(60):
    model.train()
    total_loss = 0

    for x, y1, y2 in train_loader:
        x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
        out1, out2 = model(x)
        loss = criterion(out1, y1) + criterion(out2, y2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[{epoch+1}] Loss: {total_loss:.4f}")

    # ✅ EarlyStopping 검사
    if total_loss < best_loss:
        best_loss = total_loss
        trigger = 0
        torch.save(model.state_dict(), "fall_bigru_model.pth")  # 가장 좋은 모델 저장
        print("💾 모델이 향상되어 저장되었습니다.")
    else:
        trigger += 1
        print(f"⏳ 개선 없음: {trigger}/{patience}")
        if trigger >= patience:
            print("🛑 EarlyStopping 발동 - 학습 종료")
            break

print("✅ 학습 완료 및 모델 저장 완료")
