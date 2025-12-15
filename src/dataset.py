import os
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from scipy.ndimage import zoom


class UniversalRadarDataset(Dataset):
    def __init__(self, data_dir, target_size=(64, 30)):
        """
        参数:
            data_dir: 存放 .mat 文件的文件夹路径
            target_size: 模型需要的尺寸 (Freq, Time) -> (Height, Width)
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

        # 增加一个排序，保证每次加载顺序一致
        self.file_list.sort()

        print(f"检测到 {len(self.file_list)} 个样本。目标尺寸: {target_size}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        filepath = os.path.join(self.data_dir, filename)

        # 1. 加载数据
        try:
            mat_data = sio.loadmat(filepath)
            spec = mat_data['radar_spec']
            label = mat_data['activity_id'][0][0]
        except Exception as e:
            # 遇到坏数据返回全0，防止训练中断
            print(f"Error loading {filename}: {e}")
            return torch.zeros(1, *self.target_size), 0

        # 2. 预处理 (Log + 归一化)
        spec = np.log10(np.abs(spec) + 1e-6)
        spec = (spec - spec.min()) / (spec.max() - spec.min())

        # 3. 尺寸适配 (Zoom + 防御性编程)
        curr_h, curr_w = spec.shape
        target_h, target_w = self.target_size

        scale_h = target_h / curr_h
        scale_w = target_w / curr_w

        # 初步缩放
        spec_resized = zoom(spec, (scale_h, scale_w), order=1)

        # --- 🛡️ 核心修复：画布法 (Canvas Method) ---
        # 不管 zoom 出来是多少，强制对齐到 target_size

        # A. 创建一个标准尺寸的空画布 (全0)
        canvas = np.zeros(self.target_size, dtype=np.float32)

        # B. 计算实际能贴进去的大小 (取最小值，防止溢出)
        actual_h = min(spec_resized.shape[0], target_h)
        actual_w = min(spec_resized.shape[1], target_w)

        # C. 将数据贴入画布左上角 (或者中心，视情况而定)
        # 这里的 [:actual_h, :actual_w] 既实现了裁切(Crop)，也自然处理了填充(Pad)
        canvas[:actual_h, :actual_w] = spec_resized[:actual_h, :actual_w]

        # D. 将画布作为最终结果
        final_spec = canvas

        # 4. 转 Tensor (1, H, W)
        spec_tensor = torch.FloatTensor(final_spec).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return spec_tensor, label_tensor


if __name__ == "__main__":
    # 调试逻辑
    print("正在测试 Dataset 读取逻辑...")
    # 确保路径指向你之前生成的假数据文件夹
    current_script_path = os.path.abspath(__file__)

    src_dir = os.path.dirname(current_script_path)

    project_root = os.path.dirname(src_dir)

    data_dir = os.path.join(project_root, "raw_data_external")

    if not os.path.exists(data_dir):
        print(f" 错误：找不到文件夹 {data_dir}，请先运行 make_fake_real_data.py")
    else:
        dataset = UniversalRadarDataset(data_dir=data_dir)

        # 随机抽查 5 个样本，看形状是否稳定
        print(f"\n随机抽查 5 个样本的形状 (目标: 1, 64, 30):")
        for i in range(5):
            idx = np.random.randint(0, len(dataset))
            img, lbl = dataset[idx]
            print(f"样本 {idx}: Shape={list(img.shape)} | Label={lbl}")

            # 严格断言检查
            assert img.shape == (1, 64, 30), f"尺寸错误! 期望 (1, 64, 30), 实际 {img.shape}"

        print("\n 所有检查通过！Dataset 代码非常稳健。")