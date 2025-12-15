# make_dataset.py
import numpy as np
import os
from src.simulator import RadarSimulator
from src.processor import radar_stft
from tqdm import tqdm  # 进度条库，如果没有请 pip install tqdm

# 配置
DATASET_DIR = "data"
SAMPLES_PER_CLASS = 500  # 每类生成500个，共1000个
FS = 1000


def generate_dataset():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    sim = RadarSimulator()

    # 用于存放数据和标签的列表
    data = []
    labels = []

    print(" 开始生成")

    # 1. 生成无人机数据 (Label 0)
    print("正在生成 Drone 数据...")
    for _ in tqdm(range(SAMPLES_PER_CLASS)):
        _, sig = sim.generate_signal(target_type='drone', fs=FS)
        spectrogram = radar_stft(sig, FS)
        data.append(spectrogram)
        labels.append(0)  # 0 代表 Drone

    # 2. 生成行人数据 (Label 1)
    print("正在生成 Pedestrian 数据...")
    for _ in tqdm(range(SAMPLES_PER_CLASS)):
        _, sig = sim.generate_signal(target_type='pedestrian', fs=FS)
        spectrogram = radar_stft(sig, FS)
        data.append(spectrogram)
        labels.append(1)  # 1 代表 Pedestrian

    # 3. 转换为 Numpy 数组并保存
    # X shape: (1000, 频率维度, 时间维度)
    X = np.array(data, dtype=np.float32)
    # Y shape: (1000,)
    Y = np.array(labels, dtype=np.int64)

    # 4. 数据归一化 (Normalization) - 深度学习关键一步！
    # 将数据缩放到 [0, 1] 或 [-1, 1] 之间，这里我们用简单的 Min-Max 归一化
    # 注意：在实际工程中，应该用训练集的 min/max 去归一化测试集，这里简化处理
    X_min = X.min()
    X_max = X.max()
    X_norm = (X - X_min) / (X_max - X_min)

    print(f"\n数据集生成完毕!")
    print(f"数据形状 X: {X_norm.shape}")
    print(f"标签形状 Y: {Y.shape}")

    # 保存为 .npy 文件 (二进制格式，读取极快)
    np.save(os.path.join(DATASET_DIR, "dataset_X.npy"), X_norm)
    np.save(os.path.join(DATASET_DIR, "dataset_Y.npy"), Y)
    print(f" 文件已保存至 {DATASET_DIR}/")


if __name__ == "__main__":
    generate_dataset()