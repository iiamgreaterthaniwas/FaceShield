"""
Fawkes 基线方法实现
论文：Fawkes: Protecting Privacy against Unauthorized Deep Learning Models (USENIX 2020)
原版：https://github.com/Shawn-Shan/fawkes

与本项目方法的核心区别：
  本项目 (PGD/MI-FGSM)：
    目标 → 最大化换脸模型的身份特征距离（L_id）
    约束 → L∞ 范数 + 自适应 ε
  Fawkes：
    目标 → 将特征向量"推向"一个不同身份的目标特征（feature cloaking）
    约束 → L2 范数（DSSIM 感知距离）
    优化 → L-BFGS（而非 SGD 类迭代）

本实现为 Fawkes 的简化版本（PyTorch 重实现）：
  - 随机选取 batch 内其他样本作为 target feature
  - 使用 Adam 优化器替代 L-BFGS（更快）
  - 保留 L∞ 约束（与本项目统一，便于公平对比）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FawkesBaseline:
    """
    Fawkes 基线扰动生成器

    核心策略（Feature Cloaking）：
      1. 用身份特征提取器得到原始特征 f(x)
      2. 从 batch 内随机选取另一张人脸的特征 f(x_tgt) 作为"伪装目标"
      3. 最小化 ||f(x+δ) - f(x_tgt)||²，
         使受保护的图像看起来像另一个人
      4. 约束 ||δ||∞ ≤ ε（与本项目方法使用相同约束，公平对比）

    与本项目方法的对比：
      本项目最大化 id 距离（"推开"），Fawkes 最小化到目标的距离（"推向"）
      前者无需指定 target，后者需要一个目标身份特征
    """

    def __init__(
        self,
        epsilon: float = 8 / 255,
        num_steps: int = 40,
        lr: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            epsilon:    L∞ 约束（与本项目方法保持一致）
            num_steps:  优化迭代次数
            lr:         Adam 学习率
            device:     计算设备
        """
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.lr = lr
        self.device = device

    def generate(
        self,
        img: torch.Tensor,
        target_model: nn.Module,
        target_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成 Fawkes 风格的对抗扰动（Feature Cloaking）

        Args:
            img:          原始人脸图像 [B, 3, H, W]，值域 [0, 1]
            target_model: 目标换脸模型（SimSwapWrapper 或 DeepFaceLabWrapper）
            target_feat:  目标身份特征 [B, 512]（可选，不提供则从 batch 内循环采样）

        Returns:
            adv_img:      添加扰动后的图像 [B, 3, H, W]
            perturbation: 扰动本身 [B, 3, H, W]
        """
        img = img.to(self.device)
        target_model = target_model.to(self.device).eval()

        # 确定目标特征：若未提供，从 batch 内循环位移采样（Fawkes 原文策略）
        with torch.no_grad():
            if target_feat is None:
                # 循环位移：batch 内每张图的"伪装目标"是下一张图的特征
                shifted = torch.roll(img, shifts=1, dims=0)
                target_feat = target_model.get_id_feature(shifted).detach()
            else:
                target_feat = target_feat.to(self.device).detach()

        # 初始化扰动（Fawkes 原始从 0 开始）
        delta = torch.zeros_like(img, requires_grad=False)

        # 使用 Adam 优化器（比 L-BFGS 更稳定，适合 batch 模式）
        delta = delta.requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for step in range(self.num_steps):
            optimizer.zero_grad()
            adv_img = torch.clamp(img + delta, 0, 1)

            # Fawkes 核心损失：最小化到目标特征的 L2 距离
            feat_adv = target_model.get_id_feature(adv_img)
            loss_cloak = F.mse_loss(feat_adv, target_feat)

            # 感知约束：防止图像过度失真（可选项，增强视觉质量）
            loss_percep = F.mse_loss(adv_img, img)
            loss = loss_cloak + 0.05 * loss_percep

            loss.backward()
            optimizer.step()

            # L∞ 投影（裁剪到 [-ε, ε]）
            with torch.no_grad():
                delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
                delta.data = torch.clamp(img + delta.data, 0, 1) - img

        adv_img = torch.clamp(img + delta.detach(), 0, 1)
        perturbation = adv_img - img
        return adv_img, perturbation

    def generate_single(
        self,
        img: torch.Tensor,
        target_model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单张图片生成（自动构造随机目标特征）

        Args:
            img: [1, 3, H, W] 单张图片
            target_model: 目标模型
        Returns:
            adv_img, perturbation
        """
        img = img.to(self.device)
        with torch.no_grad():
            # 对单张图片：用高斯噪声图的特征作为"目标"（模拟 Fawkes 随机目标策略）
            noise_img = torch.rand_like(img)
            target_feat = target_model.get_id_feature(noise_img)
        return self.generate(img, target_model, target_feat=target_feat)
