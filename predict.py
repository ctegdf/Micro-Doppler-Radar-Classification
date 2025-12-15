import torch
import numpy as np
from src.simulator import RadarSimulator
from src.processor import radar_stft
from src.model import RadarCNN

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_best.pth"  # 确保你刚才运行 train.py 生成了新的权重
FS = 1000
TARGET_WIDTH = 60  # 对应训练时的 2秒数据 (STFT后大约是60个时间步)


def load_trained_model():
    print(f"正在加载模型: {MODEL_PATH}")
    # 1. 先实例化基础模型
    model = RadarCNN()

    # 2. 修改全连接层结构 (此时新层在 CPU)
    import torch.nn as nn
    model.fc = nn.Sequential(
        nn.Linear(32 * 8 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    )

    model = model.to(DEVICE)

    try:
        # 加载权重
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except RuntimeError as e:
        print(f" 权重加载失败: {e}")
        print(" 提示: 请确认你已经运行过新的 train.py 并且生成了新的 model_best.pth")
        exit()

    model.eval()
    return model


def process_signal(signal):
    """
    预处理流水线：STFT -> 裁剪/填充 -> 归一化
    必须与 make_dataset.py 中的逻辑严格一致
    """
    # 1. STFT
    spectrogram = radar_stft(signal, FS)

    # 2. 尺寸对齐 (强制固定为 64x60)
    # 训练时我们取了前 60 个时间步
    if spectrogram.shape[1] > TARGET_WIDTH:
        spectrogram = spectrogram[:, :TARGET_WIDTH]
    elif spectrogram.shape[1] < TARGET_WIDTH:
        pad_width = TARGET_WIDTH - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)))

    # 3. 归一化 (使用训练集的近似统计值，或者简单的 Min-Max)
    # 这里使用简单的动态 Min-Max，实际工程中应记录训练集的 Mean/Std
    spec_min, spec_max = spectrogram.min(), spectrogram.max()
    if spec_max - spec_min > 0:
        spec_norm = (spectrogram - spec_min) / (spec_max - spec_min)
    else:
        spec_norm = spectrogram

    return spec_norm


def inference(model, signal):
    # 预处理
    spec_processed = process_signal(signal)

    # 转 Tensor: (1, 1, 64, 60)
    input_tensor = torch.FloatTensor(spec_processed).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_idx].item()

    class_names = {0: "Drone (无人机)", 1: "Human (行人)"}
    return class_names[pred_idx], confidence


if __name__ == "__main__":
    # 1. 加载模型
    model = load_trained_model()

    # 2. 实例化仿真器
    sim = RadarSimulator()

    print("\n--- 开始高级仿真信号实时测试 ---")

    # --- 测试 1: 无人机 ---
    print("\n[测试 1] 生成复杂无人机信号 (Blade Flash)...")
    # 注意：这里时间设为 2.0 以匹配训练配置
    _, sig_drone = sim.generate_signal(target_type='drone', fs=FS, duration=2.0)
    label, conf = inference(model, sig_drone)
    print(f" 预测结果: {label}")
    print(f" 置信度:   {conf * 100:.2f}%")

    # --- 测试 2: 行人 ---
    print("\n[测试 2] 生成复杂 Boulic 行人信号 (多散射点)...")
    # 修正：这里使用 'human' 而不是 'pedestrian'
    _, sig_human = sim.generate_signal(target_type='human', fs=FS, duration=2.0)
    label, conf = inference(model, sig_human)
    print(f" 预测结果: {label}")
    print(f" 置信度:   {conf * 100:.2f}%")

    # 验证
    if "Human" in label:
        print("\n 成功识别！")
    else:
        print("\n 识别错误，请检查模型训练是否充分。")