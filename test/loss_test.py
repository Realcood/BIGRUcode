import matplotlib.pyplot as plt

# 학습 루프 전에 선언했어야 함
# 예: loss_list = [2.4, 1.8, 1.2, 0.95, 0.76, ...]
# 학습 루프 안에서 매 epoch마다 total_loss 저장

loss_list = [2.34, 1.88, 1.45, 1.01, 0.89, 0.75, 0.62, 0.55]  # 예시 값

plt.figure(figsize=(8, 5))
plt.plot(loss_list, marker='o', label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("📉 Loss Curve (Fall Detection)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
