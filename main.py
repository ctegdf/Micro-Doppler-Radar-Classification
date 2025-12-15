import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from src.simulator import RadarSimulator  # 确保你建好了文件夹结构


def plot_spectrogram(t, x, fs, title):
    # TODO: 参考上次的代码，使用 signal.spectrogram 计算时频图
    # 注意：这里的 x 是复数，spectrogram会自动处理复数输入(双边谱)，
    # 但我们通常需要手动用 centering='gravity' 或者 fftshift 来调整显示，
    # 简单起见，可以先对 signal.spectrogram 的结果手动 fftshift

    f, t_spec, Sxx = signal.spectrogram(x, fs, nperseg=64, noverlap=32, return_onesided=False)

    # 频移，把0频率移到中心
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    plt.pcolormesh(t_spec, f, 10 * np.log10(np.abs(Sxx)), shading='gouraud')
    plt.title(title)
    plt.ylabel('Doppler Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Power [dB]')


# 1. 实例化仿真器
sim = RadarSimulator()
fs = 1000  # 慢时间采样率

# 2. 生成两种数据
# TODO: 调用 generate_signal 生成 drone 和 pedestrian 数据
t_drone, s_drone = sim.generate_signal(target_type='drone')  # 补全这行
t_ped, s_ped = sim.generate_signal(target_type='pedestrian')  # 补全这行

# 3. 绘图对比
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# TODO: 画出 drone 的时频图 (如果 s_drone 成功生成)
if s_drone is not None:
    plot_spectrogram(t_drone, s_drone, fs, 'Drone Micro-Doppler')

plt.subplot(1, 2, 2)
# TODO: 画出 pedestrian 的时频图
if s_ped is not None:
    plot_spectrogram(t_ped, s_ped, fs, 'Pedestrian Micro-Doppler')

plt.tight_layout()
plt.show()