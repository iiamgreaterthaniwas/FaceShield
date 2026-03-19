# 基于对抗扰动的人脸隐私保护系统 · FaceShield

> 于宙琛 · 保定学院人工智能学院 · 2026届毕业设计

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 项目简介

**FaceShield** 是一套基于对抗样本技术的人脸隐私**主动防御**系统。在用户将照片上传至社交媒体之前，预先向图像中叠加人眼难以察觉的对抗扰动（Adversarial Perturbation），干扰以 SimSwap、DeepFaceLab 为代表的深度换脸（Deepfake）模型的身份特征提取过程，从源头阻断深度伪造内容的生成。

```
原始人脸图像
      │
      ▼
  对抗扰动生成引擎
  ┌─────────────────────────────────┐
  │  算法：PGD / MI-FGSM / FGSM    │
  │  约束：L∞ ≤ ε（默认 8/255）    │
  │  策略：自适应 ε（Sobel 梯度图） │
  └─────────────────────────────────┘
      │
      ▼
  保护后图像（PSNR > 36 dB，视觉无感知变化）
      │
      ▼  上传社交媒体后若被 Deepfake 处理
  换脸失败 / 身份特征被成功干扰 ✅
```

### 与 Fawkes 的核心区别

| 维度 | 本项目（PGD） | Fawkes |
|------|-------------|--------|
| 攻击目标 | SimSwap ArcFace 换脸模型 | 人脸识别分类模型 |
| 优化方向 | 最大化身份特征余弦距离（推离） | 最小化到目标特征的距离（推向） |
| 约束范数 | L∞ | L2（DSSIM） |
| 优化方式 | PGD 多步梯度迭代 | L-BFGS / Adam |
| 自适应策略 | ✅ 基于 Sobel 梯度图动态分配 ε | ✗ |

---

## 项目结构

```
face_privacy_protection/
│
├── core/
│   ├── __init__.py
│   └── perturbation.py          # 对抗扰动生成算法（PGD / MI-FGSM / FGSM + 自适应 ε）
│
├── models/
│   ├── simswap_wrapper.py        # SimSwap 封装（白盒攻击目标，ArcFace IR-SE50）
│   ├── deepfacelab_wrapper.py    # DeepFaceLab 封装（迁移性实验目标模型）
│   └── fawkes_baseline.py        # Fawkes 基线方法复现（Feature Cloaking）
│
├── flask_ui/
│   ├── app.py                    # Flask 后端（RESTful API + SSE 实时推流）
│   ├── templates/
│   │   └── index.html            # 前端主页面（四标签页交互界面）
│   └── static/
│       └── js/
│           └── main.js           # 前端交互逻辑
│
├── utils/
│   └── dataset.py                # 数据集加载、预处理、DataLoader 工厂
│
├── evaluation/
│   └── metrics.py                # ASR / PSNR / SSIM / L∞ 评估指标
│
├── scripts/
│   ├── run_experiment.py         # 主实验脚本（参数扫描 + 算法对比 + 可视化）
│   └── prepare_dataset.py        # 数据集下载与预处理工具
│
├── data/
│   ├── raw/                      # 原始数据（CelebA-HQ 等）
│   ├── processed/
│   │   └── celeba_hq/images/     # 预处理后图像（224×224 PNG）
│   └── results/                  # 实验结果（图像、指标、热力图、折线图）
│
├── configs/
│   └── default.yaml              # 超参数配置文件
│
├── checkpoints/                  # SimSwap 预训练权重（需自行下载，见下文）
│
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：

```
torch>=1.12.0
torchvision>=0.13.0
flask>=2.0
opencv-python
Pillow
scikit-image
seaborn
matplotlib
pyyaml
```

### 2. 准备数据集

```bash
# 调试模式（生成随机占位图像，仅用于验证代码流程）
python scripts/prepare_dataset.py --mode dummy --num 50

# 使用自己的人脸图像（推荐正式实验使用）
python scripts/prepare_dataset.py --mode custom --src_dir /path/to/your/faces

# 通过 Kaggle API 下载 CelebA-HQ（需提前配置 ~/.kaggle/kaggle.json）
python scripts/prepare_dataset.py --mode kaggle

# 查看完整数据集获取指引
python scripts/prepare_dataset.py --show_guide

# 检查数据集状态
python scripts/prepare_dataset.py --mode check --output_dir data/processed/celeba_hq
```

> **注意**：正式实验建议使用 CelebA-HQ 数据集，调试模式生成的随机图像不具备实验参考价值。

### 3. 启动 Web 可视化系统

```bash
python flask_ui/app.py
# 浏览器访问 http://localhost:5000
```

系统提供四个功能标签页：

| 标签页 | 功能 |
|--------|------|
| 🛡 图像保护 | 上传人脸图像，实时生成保护图，四格对比展示（原图 / 保护图 / 扰动 / 自适应ε分布） |
| 📦 数据集准备 | 批量预处理本地图像，SSE 实时进度日志 |
| ⚗ 批量攻防实验 | ε×步数参数扫描（热力图）、Fawkes 基线对比、DFL 迁移性测试 |
| 📊 实验结果 | 折线图、热力图、算法对比柱状图一键查看 |

### 4. 命令行运行实验

```bash
# 单次实验（ε=8/255，PGD 40步，自适应ε启用）
python scripts/run_experiment.py \
    --data_dir data/processed/celeba_hq \
    --epsilon 8 --steps 40 --attack pgd \
    --num_samples 100 --save_images --adaptive

# 对比三种算法（FGSM / PGD / MI-FGSM），生成热力图与折线图
python scripts/run_experiment.py \
    --data_dir data/processed/celeba_hq \
    --compare_methods

# 指定攻击算法
python scripts/run_experiment.py --attack mifgsm --epsilon 12 --steps 20
```

---

## 核心算法

### 对抗扰动生成（PGD）

$$\delta_0 \sim \text{Uniform}(-\varepsilon, \varepsilon)$$

$$\delta_{t+1} = \Pi_{\|\delta\|_\infty \leq \varepsilon} \left( \delta_t + \alpha \cdot \text{sign}(\nabla_\delta \mathcal{L}) \right)$$

```
初始化: δ ~ Uniform(-ε, ε)          ← 随机初始化打破对称性（避免梯度死锁）
For t = 1 to T:
    adv_img = clip(x + δ, 0, 1)
    feat_adv = ArcFace(adv_img)      ← 目标模型身份特征提取
    L = -CosSim(feat_adv, feat_orig) + λ·MSE(adv_img, x)
    g_t = ∇_δ L                      ← 反向传播计算梯度
    δ = δ + α · sign(g_t)            ← 梯度符号更新
    δ = clip(δ, -ε_map, ε_map)       ← 自适应 L∞ 投影
返回: x_adv = clip(x + δ, 0, 1)
```

### 多目标损失函数

$$\mathcal{L} = \underbrace{-\text{CosSim}(f(x_{adv}), f(x))}_{\mathcal{L}_{id}\ \text{（身份欺骗）}} + \lambda \cdot \underbrace{\text{MSE}(x_{adv},\ x)}_{\mathcal{L}_{percep}\ \text{（感知保真）}}$$

- **L_id**：最大化对抗图像与原图在 ArcFace 特征空间中的余弦距离，使换脸模型无法提取正确身份
- **L_percep**：约束像素级失真，保持图像视觉质量（λ = 0.1）

### 自适应 ε 调度器（AdaptiveEpsilonScheduler）

标准 PGD 对所有像素使用统一扰动上界，忽略了人眼对不同区域的敏感度差异。本项目提出基于 Sobel 梯度图的自适应策略：

```
Step 1: 灰度转换  G = 0.299R + 0.587G + 0.114B
Step 2: Sobel 梯度  M = √(Gx² + Gy²)
Step 3: 5×5 高斯核平滑  M̃ = GaussianBlur(M)
Step 4: 归一化映射  ε_map = ε_min + (ε_max - ε_min) × normalize(M̃)
```

$$\varepsilon_{map}(i,j) = \varepsilon_{min} + (\varepsilon_{max} - \varepsilon_{min}) \cdot \tilde{W}(i,j)$$

其中默认 ε_min = 0.5ε，ε_max = 1.5ε。平滑区域（皮肤）使用更小扰动保持视觉质量，纹理丰富区域（发际线、边缘）使用更大扰动增强防御效果。

**实测效果**：在 ASR 不变的前提下，PSNR 提升约 0.5～1.0 dB，SSIM 提升约 0.004。

---

## 评估指标

| 指标 | 含义 | 目标值 | 当前结果（ε=8/255，PGD T=40） |
|------|------|--------|-------------------------------|
| **ASR** | 防御成功率（换脸被成功干扰的比例） | > 80% | **100%** ✅ |
| **PSNR** | 峰值信噪比（图像质量保持） | > 30 dB | **36～38 dB** ✅ |
| **SSIM** | 结构相似度（人眼感知质量） | > 0.95 | **0.93** ⚠️ |
| **L∞** | 最大像素扰动幅度 | ≤ ε/255 | **满足约束** ✅ |
| **处理时间** | 单张图像（GPU） | — | **2～5 s** |

> **SSIM 说明**：当前 SSIM 约 0.93，略低于目标 0.95，原因是 ε=8/255 较大。在质量优先场景下，将 ε 调低至 4/255 可同时满足两项指标，防御效果有所降低但仍可接受。

### 三种算法对比（ε=8/255，约 100 张样本）

| 算法 | ASR | PSNR (dB) | SSIM | 处理时间 |
|------|-----|-----------|------|---------|
| FGSM（Fawkes-like，单步） | 100% | 36.84 | 0.93 | ~0.3 s |
| **PGD（本系统，推荐）** | **100%** | **36.84** | **0.93** | ~3.2 s |
| MI-FGSM（动量，迁移性↑） | 100% | 34.92 | 0.91 | ~3.5 s |

---

## 配置文件说明

`configs/default.yaml` 包含所有可调参数：

```yaml
attack:
  epsilon: 8          # 扰动强度（像素值，对应 8/255）
  alpha: 1            # 每步步长（像素值，对应 1/255）
  num_steps: 40       # PGD 迭代次数
  attack_type: "pgd"  # 算法选择：pgd / mifgsm / fgsm
  momentum: 1.0       # MI-FGSM 动量系数

loss:
  lambda_id: 1.0      # 身份欺骗损失权重
  lambda_percep: 0.1  # 感知质量损失权重

eval:
  asr_threshold: 0.5  # ASR 判定阈值（余弦相似度）
  target_psnr: 30.0
  target_ssim: 0.95
```

---

## 获取 SimSwap 预训练权重

本项目以 SimSwap 的 ArcFace 身份编码器（IR-SE50）为白盒攻击目标，需要预训练权重：

**方式 A：官方仓库**
```bash
git clone https://github.com/neuralchen/SimSwap
# 按照 SimSwap 官方说明下载预训练权重
# 将 checkpoints/ 目录复制到本项目根目录
```

**方式 B：无权重调试模式**

代码已设计为在无权重时自动使用随机初始化，可用于验证完整代码流程。此模式下 ASR 指标无实际意义，但 PSNR / SSIM 计算正常。

---

## 已知问题与后续计划

### 当前局限性

- **样本量**：当前测试集约 100 张，统计结论存在一定波动，待扩充至 200 张以上后重新验证
- **DFL 迁移实验**：DeepFaceLab 代理模型目前使用随机初始化权重，实验结论参考价值有限，待补充真实预训练权重
- **压缩鲁棒性**：图像经社交媒体 JPEG 压缩后扰动可能被削弱，待引入可微分 JPEG 近似层进行增强

### 后续研究方向

1. **压缩鲁棒性**：引入可微分 JPEG 层，使扰动在压缩后仍保持有效性
2. **跨模型迁移**：集成输入多样化（Input Diversity）、TI-FGSM 等迁移增强技术
3. **通用对抗扰动（UAP）**：训练与输入无关的全局扰动，将保护时间从秒级降至毫秒级
4. **扩散模型防御**：探索针对 InstructPix2Pix、IP-Adapter 等新型生成模型的防御策略

---

## 参考文献

- **[SimSwap]** Chen R, et al. *SimSwap: An Efficient Framework For High Fidelity Face Swapping*. ACM MM 2020.
- **[DeepFaceLab]** Perov I, et al. *DeepFaceLab: Integrated, flexible and extensible face-swapping framework*. arXiv 2020.
- **[Fawkes]** Shan S, et al. *Fawkes: Protecting Privacy against Unauthorized Deep Learning Models*. USENIX Security 2020.
- **[PGD]** Madry A, et al. *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR 2018.
- **[MI-FGSM]** Dong Y, et al. *Boosting Adversarial Attacks with Momentum*. CVPR 2018.
- **[FGSM]** Goodfellow I J, et al. *Explaining and Harnessing Adversarial Examples*. ICLR 2015.

---

## 致谢

感谢指导教师李宝才讲师与章飞高级工程师的悉心指导。本项目使用了 SimSwap 和 DeepFaceLab 的开源代码及 CelebA-HQ 数据集，在此向相关研究者致谢。