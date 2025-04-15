import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import os
from model import FallBiGRUAttentionNet

# ✅ 테스트용 데이터셋 클래스 (학습용과 동일)
class FallDataset(torch.utils.data.Dataset):
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

# ✅ 전체 데이터셋 로드
dataset = FallDataset("pose_data")
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 학습된 모델 로드
model = FallBiGRUAttentionNet(input_dim=24, hidden_dim=128, num_layers=2).to(device)
model.load_state_dict(torch.load("fall_bigru_model.pth", map_location=device))
model.eval()

# ✅ 예측 수행
y_true = []
y_pred = []

with torch.no_grad():
    for x, y1, y2 in test_loader:
        x = x.to(device)
        out1, _ = model(x)
        pred = torch.argmax(out1, dim=1)

        y_true.extend(y1.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

# ✅ 혼동 행렬 생성 및 시각화
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["정상", "낙상"])
disp.plot(cmap="Blues", values_format="d")
plt.title("🧠 Fall Detection Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()
