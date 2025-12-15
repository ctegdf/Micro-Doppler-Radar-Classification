import numpy as np


class RadarSimulator:
    """
    雷达回波信号仿真器
    用于生成具有微多普勒效应的雷达基带信号
    """

    def __init__(self, lambda_val=0.03):
        # 默认波长 0.03m (对应 10GHz X波段雷达)
        self.lambda_val = lambda_val

    def generate_signal(self, duration=1.0, fs=1000, target_type='drone'):
        """
        生成单个目标的时域回波信号

        参数:
            duration: 信号时长 (秒)
            fs: 采样率 (Hz) - 这里的采样率是指慢时间(Slow-time)采样率，即PRF
            target_type: 'drone' (无人机) 或 'pedestrian' (行人)

        返回:
            t: 时间轴
            signal: 复数基带信号
        """
        t = np.arange(0, duration, 1 / fs)

        # 定义目标的运动模型 R(t)
        if target_type == 'drone':
            # 无人机特点：悬停或慢速移动(v小)，旋翼转速极快(f_vib高)，振幅极小(A_vib小)
            # 模拟：主旋翼的高频振动
            R0 = 10.0  # 距离10米
            v = 0.5  # 速度 0.5 m/s
            A_vib = 0.005  # 振幅 0.5 cm
            f_vib = 100.0  # 振动频率 100 Hz (高频)

            # 距离公式
            R_t = R0 + v * t + A_vib * np.sin(2 * np.pi * f_vib * t)

        elif target_type == 'pedestrian':
            # 行人特点：速度适中，摆臂频率低(f_vib低)，幅度大(A_vib大)
            R0 = 10.0
            v = 1.5  # 速度 1.5 m/s
            A_vib = 0.3  # 摆臂幅度 30 cm
            f_vib = 2.0  # 摆臂频率 2 Hz (低频)

            R_t = R0 + v * t + A_vib * np.sin(2 * np.pi * f_vib * t)

        else:
            raise ValueError("Unknown target type")

        # 核心物理公式：相位 phi = 4 * pi * R / lambda
        phase = (4 * np.pi * R_t) / self.lambda_val

        # 生成复数信号 (加上一点高斯白噪声)
        noise = 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        signal = np.exp(1j * phase) + noise

        return t, signal


# 调试代码：只有直接运行此文件时才会执行
if __name__ == "__main__":
    print("正在测试 RadarSimulator...")
    sim = RadarSimulator()
    t, s = sim.generate_signal(target_type='drone')
    print(f"生成信号长度: {len(s)}, 数据类型: {s.dtype}")
    print("测试通过！")