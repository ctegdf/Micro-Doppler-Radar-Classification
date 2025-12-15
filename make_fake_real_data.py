import numpy as np
import scipy.io as sio  # 需要 pip install scipy
import os

# 模拟一个外部的、混乱的真实数据集文件夹
RAW_DATA_DIR = "raw_data_external"


def create_messy_data():
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)

    print(f"正在生成模拟的'外部真实数据'到 {RAW_DATA_DIR}...")

    # 模拟 100 个样本，尺寸不再是整齐的 64x30，而是随机的巨大尺寸
    # 模拟真实雷达：采样率更高，时间更长
    for i in range(100):
        # 随机尺寸：高度 100~200，宽度 200~400 (比我们的模型大很多)
        h = np.random.randint(100, 200)
        w = np.random.randint(200, 400)

        # 生成随机频谱数据
        spectrogram = np.random.randn(h, w).astype(np.float32)

        # 随机标签 (0~5)
        label = np.random.randint(0, 6)

        # 保存为 .mat 文件 (模拟 MATLAB 数据源)
        # 文件名也很乱
        filename = f"recording_{i:04d}_class_{label}.mat"
        filepath = os.path.join(RAW_DATA_DIR, filename)

        # MATLAB 文件通常是一个字典
        sio.savemat(filepath, {"radar_spec": spectrogram, "activity_id": label})

    print(" 数据生成完毕。")


if __name__ == "__main__":
    create_messy_data()