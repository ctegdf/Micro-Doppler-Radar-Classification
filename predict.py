import torch
import numpy as np
from src import RadarSimulator, RadarCNN, radar_stft

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_best.pth"
FS = 1000


def load_trained_model():
    print(f"正在加载模型: {MODEL_PATH}")
    model = RadarCNN().to(DEVICE)

    # 加载权重
    # map_location确保即使在只有CPU的机器上也能加载GPU训练的模型
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model.eval()  # 极其重要！切换到评估模式 (固定 BatchNorm 和 Dropout)
    return model


def inference(model, signal):
    """
    对单个信号进行推理
    """
    # 1. 预处理 (STFT)
    # output shape: (64, 30)
    spectrogram = radar_stft(signal, FS)

    # 2. 归一化 (这里简化处理，使用理论最大最小值，实际应使用训练集的统计值)
    # 假设之前的 Log 谱大概在 -80 到 0 dB 之间
    spec_norm = (spectrogram - (-80)) / (0 - (-80))

    # 3. 维度调整
    # numpy (64, 30) -> tensor (1, 1, 64, 30)
    input_tensor = torch.FloatTensor(spec_norm).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    # 4. 前向传播
    with torch.no_grad():  # 推理不需要计算梯度
        output = model(input_tensor)
        # output 是 logits，通过 softmax 获取概率
        probabilities = torch.nn.functional.softmax(output, dim=1)

        # 获取最大概率的类别索引
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_idx].item()

    class_names = {0: "Drone (无人机)", 1: "Pedestrian (行人)"}
    return class_names[pred_idx], confidence


if __name__ == "__main__":
    # 1. 加载模型
    model = load_trained_model()

    # 2. 实例化仿真器 (模拟实时雷达数据流)
    sim = RadarSimulator()

    print("\n--- 开始实时测试 ---")

    # 测试案例 A: 生成一个无人机信号
    print("\n[测试 1] 正在生成真实 Drone 信号...")
    _, sig_drone = sim.generate_signal(target_type='drone', fs=FS)
    label, conf = inference(model, sig_drone)
    print(f"模型预测: {label}")
    print(f"置信度:   {conf * 100:.2f}%")

    # 测试案例 B: 生成一个行人信号
    print("\n[测试 2] 正在生成真实 Pedestrian 信号...")
    _, sig_ped = sim.generate_signal(target_type='pedestrian', fs=FS)
    label, conf = inference(model, sig_ped)
    print(f"模型预测: {label}")
    print(f"置信度:   {conf * 100:.2f}%")

    # 验证是否正确
    if label == "Pedestrian (行人)" and conf > 0.9:
        print("\n 推理功能验证成功！")
    else:
        print("\n 推理结果异常，请检查数据预处理流程。")