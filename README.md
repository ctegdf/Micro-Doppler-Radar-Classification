# 📡 Micro-Doppler Radar Classification with CNN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Deep Learning based Micro-Doppler Signature Classification for Radar Systems**
> 
> *Implementing Boulic Human Motion Model & Drone Blade Flash Simulation*

##  项目简介 (Introduction)

本项目实现了一个端到端的雷达信号处理与深度学习分类系统，专注于 **微多普勒 (Micro-Doppler)** 特征的提取与识别。不同于简单的正弦波仿真，本项目内置了高保真的物理仿真器：

* **Human**: 基于 **Boulic Human Motion Model**，通过多散射点（躯干、四肢）相干叠加，模拟真实行人的肢体摆动特征。
* **Drone**: 模拟多旋翼无人机的 **Blade Flash** 效应及高频微动特征。

模型在 STFT 时频图上训练，能够以极高的置信度 (>99%) 区分复杂人体运动与无人机目标。

##  核心模块 (Core Modules)

* `src/simulator.py`: **物理仿真引擎**。实现了 RCS 加权的 5 点人体模型 (Torso + Arms + Legs) 与旋翼无人机模型。
* `src/processor.py`: **信号处理前端**。STFT (Short-Time Fourier Transform) 与对数尺度归一化。
* `src/model.py`: **轻量级 CNN**。针对雷达时频图优化的 3 层卷积神经网络。
* `src/dataset.py`: **自适应数据管道**。支持 `.npy` 序列数据及自动维度对齐 (Auto-Padding/Cropping)。

##  效果展示 (Performance)

| **Drone Micro-Doppler** | **Human Micro-Doppler (Boulic)** |
| :---: | :---: |
| <img src="docs/drone_spec.png" width="400" /> | <img src="docs/ped_spec.png" width="400" /> |
| *特征：清晰的旋翼闪烁 (Blade Flash) 边带* | *特征：复杂的肢体摆动包络 (Torso + Limbs)* |

##  快速开始 (Quick Start)

### 1. 安装依赖
```bash
pip install numpy matplotlib scipy torch tqdm
```
### 2. 生成数据集运行数据生成脚本，使用内置物理引擎生成 1200 个复杂样本：：
```bash
python make_dataset.py
```
### 3. 训练模型训练 CNN 分类器（默认 20 Epochs）
```bash
python train.py
# 训练 3层 CNN，通常在 5 个 Epoch 内收敛至 99% Acc
```

### 4. 实时推理加载训练好的模型，对新生成的随机信号进行预测：
```bash
python predict.py
```
## 技术栈 (Tech Stack)
### Signal Processing: STFT, Doppler Processing

###  Kinematic Modeling (Boulic), RCS Modeling

### Deep Learning: PyTorch, CNN

## 待办事项 (To-Do)
[x] 引入 Boulic 人体运动模型

[x] 实现训练/推理维度的自动对齐

[ ] 扩展至真实车载雷达数据集 (CARRADA/RodNet)

[ ] 部署至 Jetson Nano 边缘端