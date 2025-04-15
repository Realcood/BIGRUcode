import matplotlib.pyplot as plt
import numpy as np

# ✅ 1개 시퀀스에 대한 모델의 낙상 예측 결과 예시 (프레임당 0 또는 1)
sequence_pred = np.array([
    0, 0, 0, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
])  # 30프레임 시퀀스 예시

plt.figure(figsize=(10, 2))
plt.plot(sequence_pred, marker='o', linestyle='-', color='red', label="예측된 낙상")
plt.yticks([0, 1], ["정상", "낙상"])
plt.xlabel("프레임 순서")
plt.title("⏱️ 프레임별 낙상 감지 결과")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
