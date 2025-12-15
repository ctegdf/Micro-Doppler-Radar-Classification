import numpy as np
import os
from src.simulator import RadarSimulator
from src.processor import radar_stft
from tqdm import tqdm

DATASET_DIR = "data"
# 增加样本量，因为现在数据变复杂了，模型需要更多数据来学习
SAMPLES_PER_CLASS = 600
FS = 1000


def generate_dataset():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    sim = RadarSimulator()

    data = []
    labels = []

    print("🚀 开始生成基于 Boulic 模型的高级仿真数据...")

    # 1. 生成无人机 (Label 0)
    print("正在生成 Drone 数据 (复杂旋翼模型)...")
    for _ in tqdm(range(SAMPLES_PER_CLASS)):
        _, sig = sim.generate_signal(target_type='drone', fs=FS, duration=2.0)
        # 注意：这里我们生成 2秒 数据，STFT后尺寸会变大
        spectrogram = radar_stft(sig, FS)

        # 简单裁剪或缩放以保证尺寸一致性 (这里取前 60 个时间步)
        # 假设 STFT 输出是 (64, T)，我们取 (64, 60)
        spectrogram = spectrogram[:, :60]
        if spectrogram.shape[1] < 60:
            # 如果不够长，补零 (Padding)
            pad_width = 60 - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)))

        data.append(spectrogram)
        labels.append(0)

    # 2. 生成人体 (Label 1)
    print("正在生成 Human 数据 (多散射点模型)...")
    for _ in tqdm(range(SAMPLES_PER_CLASS)):
        _, sig = sim.generate_signal(target_type='human', fs=FS, duration=2.0)
        spectrogram = radar_stft(sig, FS)

        # 同样处理尺寸
        spectrogram = spectrogram[:, :60]
        if spectrogram.shape[1] < 60:
            pad_width = 60 - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)))

        data.append(spectrogram)
        labels.append(1)

    # 保存
    X = np.array(data, dtype=np.float32)
    Y = np.array(labels, dtype=np.int64)

    # 归一化
    X = (X - X.min()) / (X.max() - X.min())

    print(f"\n数据集生成完毕!")
    print(f"数据形状 X: {X.shape} ")
    np.save(os.path.join(DATASET_DIR, "dataset_X.npy"), X)
    np.save(os.path.join(DATASET_DIR, "dataset_Y.npy"), Y)


if __name__ == "__main__":
    generate_dataset()