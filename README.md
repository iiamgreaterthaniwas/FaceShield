# 基于对抗扰动的人脸隐私保护系统

> 于宙琛 · 保定学院人工智能学院 · 2025届毕业设计

## 项目简介

本系统通过在人脸图像上叠加**人眼难以察觉的对抗扰动**，干扰以 SimSwap 为代表的深度伪造（Deepfake）换脸模型的特征提取与生成过程，实现"先发制人"的主动隐私保护。

```
原始人脸图像
      │
      ▼
  对抗扰动生成 (PGD / MI-FGSM)
      │ L∞ ≤ ε 约束
      ▼
  保护后图像（视觉无感知变化）
      │
      ▼ 上传社交媒体后若被 Deepfake 处理
  换脸失败 / 伪造结果严重失真 ✅
```

---

## 项目结构

```
face_privacy_protection/
│
├── core/
│   ├── __init__.py
│   └── perturbation.py        # 对抗扰动生成算法（PGD / MI-FGSM / FGSM）
│
├── models/
│   └── simswap_wrapper.py     # SimSwap 换脸模型封装（白盒攻击目标）
│
├── ui/
│   └── app.py                 # Gradio 可视化交互界面
│
├── utils/
│   └── dataset.py             # 数据集加载、预处理工具
│
├── evaluation/
│   └── metrics.py             # ASR / PSNR / SSIM 评估指标
│
├── scripts/
│   ├── run_experiment.py      # 主实验脚本（批量攻防测试）
│   └── prepare_dataset.py     # 数据集下载与预处理
│
├── data/
│   ├── raw/                   # 原始数据（CelebA-HQ 等）
│   ├── processed/
│   │   └── celeba_hq/images/  # 预处理后图像（256×256 PNG）
│   └── results/               # 实验结果（图像、指标、图表）
│
├── configs/
│   └── default.yaml           # 超参数配置文件
│
├── notebooks/                 # Jupyter 分析笔记本（可选）
│
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

```bash
# 调试模式（生成随机占位图像，验证代码流程）
python scripts/prepare_dataset.py --mode dummy --num 50

# 使用自己的人脸图像（推荐）
python scripts/prepare_dataset.py --mode custom --src_dir /path/to/your/faces

# 查看数据集下载完整指引
python scripts/prepare_dataset.py --show_guide
```

### 3. 启动 Web 界面

```bash
python ui/app.py
# 浏览器访问 http://localhost:7860
```

### 4. 运行实验（命令行）

```bash
# 单次实验（ε=8, PGD 40步）
python scripts/run_experiment.py \
    --data_dir data/processed/celeba_hq \
    --epsilon 8 --steps 40 --attack pgd \
    --num_samples 100 --save_images

# 对比不同扰动强度（生成论文图表）
python scripts/run_experiment.py \
    --data_dir data/processed/celeba_hq \
    --compare_methods
```

---

## 核心算法

### 对抗扰动生成（PGD）

```
初始化: δ ~ Uniform(-ε, ε)
For t = 1 to T:
    g_t = ∇_δ L(f(x+δ), x)       # 计算梯度
    δ = δ + α · sign(g_t)         # 梯度符号更新
    δ = Proj_{‖δ‖∞ ≤ ε}(δ)       # L∞ 投影约束
返回: x_adv = clip(x + δ, 0, 1)
```

### 多目标损失函数

```
L = L_id + λ₁·L_percep
  = -CosSim(f(x_adv), f(x)) + λ₁·MSE(x_adv, x)

L_id:    最大化与原始身份特征的距离（欺骗换脸模型）
L_percep: 最小化像素差异（保持视觉质量）
```

---

## 评估指标

| 指标 | 含义 | 目标值 |
|------|------|--------|
| **ASR** | 防御成功率（换脸被干扰的比例） | > 80% |
| **PSNR** | 峰值信噪比（图像质量保持） | > 30 dB |
| **SSIM** | 结构相似度（人眼感知质量） | > 0.95 |
| **L∞** | 最大像素扰动 | ≤ ε/255 |

---

## 获取 SimSwap 预训练权重

本项目需要 SimSwap 预训练模型权重，有两种方式：

**方式 A：官方仓库**
```bash
git clone https://github.com/neuralchen/SimSwap
# 按照 SimSwap 官方说明下载预训练权重
# 将 checkpoints/ 目录复制到本项目根目录
```

**方式 B：不使用权重（调试模式）**  
代码已设计为在无权重时自动使用随机初始化，用于验证代码流程（ASR 指标无意义，但 PSNR/SSIM 计算正常）。

---

## 参考文献

- [SimSwap] Chen R, et al. "SimSwap: An Efficient Framework For High Fidelity Face Swapping" (2020)
- [Fawkes] Shan S, et al. "Fawkes: Protecting Privacy against Unauthorized Deep Learning Models" (2020)
- [PGD] Madry A, et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018)
- [MI-FGSM] Dong Y, et al. "Boosting Adversarial Attacks with Momentum" (2018)