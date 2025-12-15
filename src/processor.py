import numpy as np
from scipy import signal


def radar_stft(x, fs, nperseg=64, noverlap=32):
    """
    对雷达信号进行 STFT 处理，并转换为对数尺度的能量谱
    """
    # 1. 计算 STFT
    f, t, Zxx = signal.spectrogram(x, fs, nperseg=nperseg, noverlap=noverlap, return_onesided=False)

    # 2. 频移 (把 0Hz 移到中心)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    # 3. 取模值的平方 (能量)
    Sxx = np.abs(Zxx) ** 2

    # 4. 转 Log 尺度 (模拟雷达的 dB 处理，非常重要，否则神经网络很难训练)
    # 加上一个极小值 eps 防止 log(0)
    Sxx_log = 10 * np.log10(Sxx + 1e-9)

    return Sxx_log