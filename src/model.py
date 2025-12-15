import torch
import torch.nn as nn


class RadarCNN(nn.Module):
    def __init__(self):
        super(RadarCNN, self).__init__()

        # -----------------------------------
        # 特征提取层 (Feature Extractor)
        # -----------------------------------

        # 第一层卷积:
        # 输入: (Batch, 1, 64, 30) -> 输出: (Batch, 8, 64, 30)
        # Padding=1 保证长宽不变 (或者略微变化，取决于卷积核)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),  # 批归一化，加速收敛
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2)  # 池化: 高宽减半 -> (Batch, 8, 32, 15)
        )

        # 第二层卷积:
        # 输入: (Batch, 8, 32, 15) -> 输出: (Batch, 16, 32, 15)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 池化: 高宽再减半 -> (Batch, 16, 16, 7)
            # 注意：15/2 = 7.5 -> 向下取整为 7
        )

        # 第三层卷积:
        # 输入: (Batch, 16, 16, 7) -> 输出: (Batch, 32, 16, 7)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 池化: 高宽再减半 -> (Batch, 32, 8, 3)
        )

        # -----------------------------------
        # 分类器层 (Classifier)
        # -----------------------------------

        # Flatten: 把多维张量展平为一维向量
        # 输入维度: 32 * 8 * 3 = 768
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(128, 2)  # 输出 2 类 (Drone vs Pedestrian)
        )

    def forward(self, x):
        # x shape: (Batch, 1, 64, 30)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # 展平: (Batch, 32, 8, 3) -> (Batch, 768)
        out = out.reshape(out.size(0), -1)

        out = self.fc(out)
        return out


# ---------------------------------------------------------
# 专家级调试技巧: "Dry Run" (干跑)
# 写完模型不要直接去训练，先给个假数据看看能不能跑通！
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. 实例化模型
    model = RadarCNN()
    print("模型结构已加载。")

    # 2. 生成一个假数据 (Batch=1, Channel=1, H=64, W=30)
    # 模拟真实数据的输入尺寸
    dummy_input = torch.randn(1, 1, 64, 30)

    # 3. 试运行
    try:
        output = model(dummy_input)
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")  # 应该是 (1, 2)
        print(" 模型自检通过！维度匹配完美。")
    except Exception as e:
        print(f" 模型报错: {e}")
        print("请检查 fc 层的输入维度是否计算正确 (32 * 8 * 3)。")