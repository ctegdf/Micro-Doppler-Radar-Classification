import numpy as np


class RadarSimulator:
    """
    高级雷达仿真器：基于 Boulic 人体行走模型
    模拟多散射点（躯干+四肢）的微多普勒效应
    """

    def __init__(self, lambda_val=0.03):
        self.lambda_val = lambda_val  # 波长 0.03m (10GHz)

    def generate_signal(self, duration=2.0, fs=1000, target_type='human'):
        """
        生成复杂目标的雷达回波
        """
        t = np.arange(0, duration, 1 / fs)

        if target_type == 'drone':
            # --- 无人机模型 (多旋翼高频调制) ---
            # 基础距离运动
            R_body = 10.0 + 1.0 * t

            # 旋翼叶片闪烁 (Blade Flash)
            # 模拟 4 个旋翼，转速不同
            signal = np.zeros_like(t, dtype=np.complex128)

            # 机身回波 (强)
            signal += 1.0 * np.exp(1j * 4 * np.pi * R_body / self.lambda_val)

            # 旋翼回波 (弱，高频)
            blade_freqs = [150, 155, 145, 160]  # 不同转速
            for f in blade_freqs:
                # 调制项：模拟叶片旋转带来的周期性闪烁
                blade_mod = 0.1 * np.cos(2 * np.pi * f * t)
                signal += blade_mod * np.exp(1j * 4 * np.pi * R_body / self.lambda_val)

            # 添加强噪声
            noise = 0.5 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
            return t, signal + noise

        elif target_type == 'human':
            # --- 复杂人体模型 (Boulic Model 简化版) ---
            # 模拟 5 个散射点：躯干(Torso), 左臂, 右臂, 左腿, 右腿

            # 1. 基础参数
            v_walk = 1.3 + np.random.normal(0, 0.1)  # 行走速度 (1.3 m/s 左右)
            R0 = 10.0  # 初始距离
            cycle_freq = v_walk / 1.5  # 步频

            # 2. 各部位的多普勒特征
            # 躯干：主要是匀速运动，带一点点起伏
            v_torso = v_walk

            # 手臂/腿：在躯干速度基础上叠加大幅度摆动
            # 使用简单的正弦模型拟合 Boulic 摆动
            swing_amp_arm = 0.6 * v_walk
            swing_amp_leg = 1.0 * v_walk

            # 3. 构建回波 (相干叠加)
            signal = np.zeros_like(t, dtype=np.complex128)

            # A. 躯干回波 (能量最强，RCS=1.0)
            R_torso = R0 + v_torso * t
            signal += 1.0 * np.exp(1j * 4 * np.pi * R_torso / self.lambda_val)

            # B. 腿部回波 (能量中等 RCS=0.5，摆动大)
            # 左腿
            v_l_leg = v_torso + swing_amp_leg * np.cos(2 * np.pi * cycle_freq * t)
            R_l_leg = R0 + np.cumsum(v_l_leg) / fs  # 积分得到距离
            signal += 0.5 * np.exp(1j * 4 * np.pi * R_l_leg / self.lambda_val)

            # 右腿 (相位差 180度)
            v_r_leg = v_torso + swing_amp_leg * np.cos(2 * np.pi * cycle_freq * t + np.pi)
            R_r_leg = R0 + np.cumsum(v_r_leg) / fs
            signal += 0.5 * np.exp(1j * 4 * np.pi * R_r_leg / self.lambda_val)

            # C. 手臂回波 (能量弱 RCS=0.3，反向摆动)
            # 左臂 (与左腿反相)
            v_l_arm = v_torso + swing_amp_arm * np.cos(2 * np.pi * cycle_freq * t + np.pi)
            R_l_arm = R0 + np.cumsum(v_l_arm) / fs
            signal += 0.3 * np.exp(1j * 4 * np.pi * R_l_arm / self.lambda_val)

            # 右臂
            v_r_arm = v_torso + swing_amp_arm * np.cos(2 * np.pi * cycle_freq * t)
            R_r_arm = R0 + np.cumsum(v_r_arm) / fs
            signal += 0.3 * np.exp(1j * 4 * np.pi * R_r_arm / self.lambda_val)

            # 4. 环境杂波与噪声
            noise = 0.2 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
            return t, signal + noise

        else:
            raise ValueError(f"Unknown target: {target_type}")


if __name__ == "__main__":
    # 简单的自测
    sim = RadarSimulator()
    t, s = sim.generate_signal(target_type='human')
    print(f"生成的复杂人体信号: {s.shape}, dtype={s.dtype}")