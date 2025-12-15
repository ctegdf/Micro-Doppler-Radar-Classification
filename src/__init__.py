# 显式暴露核心类和函数，简化外部调用
from .simulator import RadarSimulator
from .processor import radar_stft
from .model import RadarCNN

# 定义包的元数据
__all__ = [
    "RadarSimulator",
    "radar_stft",
    "RadarCNN"
]

__version__ = "1.0.0"
__author__ = "ctegdf"