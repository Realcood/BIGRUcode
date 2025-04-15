import os
import pandas as pd
from collections import Counter

DATA_DIR = "pose_data"

def count_csv_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]

def analyze_label_distribution(file_list):
    label_counter = Counter()
    part_counter = Counter()

    for file in file_list:
        try:
            df = pd.read_csv(file, header=None)
            label = int(df.iloc[0, -2])  # LABEL (낙상 여부)
            part = int(df.iloc[0, -1])   # PART (낙상 부위)
            label_counter[label] += 1
            if label == 1:
                part_counter[part] += 1
        except Exception as e:
            print(f"⚠️ {file} 분석 중 오류 발생: {e}")

    return label_counter, part_counter

# 부위 매핑
PART_MAP = {0: "머리", 1: "손목", 2: "골반", 3: "기타"}

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print("❌ pose_data 폴더가 존재하지 않습니다.")
    else:
        files = count_csv_files(DATA_DIR)
        print(f"📦 총 학습 데이터 수: {len(files)}개 시퀀스 (CSV 파일)")

        label_counter, part_counter = analyze_label_distribution(files)

        print("\n📊 낙상 여부 분포:")
        print(f"   - 정상: {label_counter[0]}개")
        print(f"   - 낙상: {label_counter[1]}개")

        print("\n🦿 낙상 부위 분포 (낙상인 경우만):")
        for part, count in part_counter.items():
            print(f"   - {PART_MAP.get(part, f'Unknown({part})')}: {count}개")
