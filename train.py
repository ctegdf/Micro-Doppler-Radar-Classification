import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src import RadarSimulator, RadarCNN, radar_stft

# 全局配置
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    """
    加载数据，打乱，划分训练集/测试集，并转换为 Tensor
    """
    print(f"正在加载数据 (使用设备: {DEVICE})...")

    # 1. 读取 numpy 文件
    X = np.load("data/dataset_X.npy")  # shape: (1000, 64, 30)
    Y = np.load("data/dataset_Y.npy")  # shape: (1000,)

    # 2. 数据打乱 (Shuffle)
    # 生成随机索引
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # 3. 转换为 Tensor 并增加 Channel 维度
    # numpy (N, H, W) -> tensor (N, 1, H, W)
    X_tensor = torch.FloatTensor(X).unsqueeze(1)
    Y_tensor = torch.LongTensor(Y)

    # 4. 划分训练集 (80%) 和 测试集 (20%)
    split_idx = int(len(X) * 0.8)

    train_dataset = TensorDataset(X_tensor[:split_idx], Y_tensor[:split_idx])
    test_dataset = TensorDataset(X_tensor[split_idx:], Y_tensor[split_idx:])

    # 5. 封装为 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train():
    # 1. 准备数据
    train_loader, test_loader = load_data()

    # 2. 初始化模型、损失函数、优化器
    model = RadarCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("开始训练...")

    for epoch in range(EPOCHS):
        model.train()  # 切换到训练模式 (启用 Dropout/BatchNorm 更新)
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # --- 标准 PyTorch 训练四步走 ---
            optimizer.zero_grad()  # 1. 梯度清零
            outputs = model(inputs)  # 2. 前向传播
            loss = criterion(outputs, labels)  # 3. 计算损失
            loss.backward()  # 4. 反向传播
            optimizer.step()  # 5. 参数更新

            running_loss += loss.item()

        # 每个 Epoch 结束后评估一次测试集准确率
        acc = evaluate(model, test_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {running_loss / len(train_loader):.4f} | Test Acc: {acc:.2f}%")

    # 保存模型权重
    torch.save(model.state_dict(), "model_best.pth")
    print("模型已保存为 model_best.pth")


def evaluate(model, loader):
    """
    计算验证集准确率
    """
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度，节省显存
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)

            # 获取预测结果: max 返回 (value, index)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


if __name__ == "__main__":
    train()