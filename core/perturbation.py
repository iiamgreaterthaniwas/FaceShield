"""
对抗扰动生成核心模块
基于梯度迭代的对抗扰动生成算法，支持 PGD / MI-FGSM / FGSM 方法
目标模型：SimSwap 换脸模型

新增：自适应扰动强度策略（Adaptive Epsilon）
  - 根据图像局部复杂度动态分配扰动预算
  - 平滑区域（皮肤）给予更小扰动保持视觉质量
  - 纹理丰富区域（头发、背景边缘）给予更大扰动增强防御效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ─────────────────────────────────────────────
#  自适应 ε 策略模块
# ─────────────────────────────────────────────

class AdaptiveEpsilonScheduler:
    """
    自适应扰动强度调度器

    策略：根据图像的局部梯度幅度（图像复杂度）动态调整每个像素位置的扰动上界。

    原理：
      - 计算图像的 Sobel 梯度图作为复杂度权重 W
      - 归一化 W 到 [ε_min, ε_max]
      - 在纹理丰富处（大梯度）允许更大扰动 → 防御更强
      - 在平滑区域（小梯度）限制扰动   → 视觉质量更好

    公式:
      ε_map = ε_min + (ε_max - ε_min) * normalize(Sobel(I))
    """

    def __init__(
        self,
        epsilon_base: float = 8 / 255,
        epsilon_min_ratio: float = 0.5,
        epsilon_max_ratio: float = 1.5,
        smooth_kernel: int = 5,
    ):
        self.epsilon_base = epsilon_base
        self.epsilon_min = epsilon_base * epsilon_min_ratio
        self.epsilon_max = epsilon_base * epsilon_max_ratio
        self.smooth_kernel = smooth_kernel

    def compute_epsilon_map(self, img: torch.Tensor) -> torch.Tensor:
        """
        计算每像素的自适应扰动上界 ε_map

        Args:
            img: [B, C, H, W] 输入图像，值域 [0, 1]
        Returns:
            epsilon_map: [B, 1, H, W] 每像素扰动上界
        """
        # 转灰度计算梯度
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

        # Sobel 算子
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=img.dtype, device=img.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=img.dtype, device=img.device
        ).view(1, 1, 3, 3)

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        gradient_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        # 高斯平滑，减少边界突变
        if self.smooth_kernel > 1:
            k = self.smooth_kernel
            sigma = k / 3.0
            coords = torch.arange(k, dtype=img.dtype, device=img.device) - k // 2
            g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
            g1d = g1d / g1d.sum()
            g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)
            gauss_kernel = g2d.view(1, 1, k, k)
            gradient_mag = F.conv2d(gradient_mag, gauss_kernel, padding=k // 2)

        # 归一化到 [0, 1]
        b = gradient_mag.shape[0]
        g_flat = gradient_mag.view(b, -1)
        g_min = g_flat.min(dim=1)[0].view(b, 1, 1, 1)
        g_max = g_flat.max(dim=1)[0].view(b, 1, 1, 1)
        weight = (gradient_mag - g_min) / (g_max - g_min + 1e-8)

        epsilon_map = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * weight
        return epsilon_map.detach()

    def project_with_map(
        self, delta: torch.Tensor, epsilon_map: torch.Tensor
    ) -> torch.Tensor:
        """使用自适应 ε_map 做 L∞ 投影（替代统一 ε）
        注：torch.clamp 的 min/max 不支持 Tensor（PyTorch < 1.9），
            用 torch.max + torch.min 实现等价操作。
        """
        return torch.max(torch.min(delta, epsilon_map), -epsilon_map)


# ─────────────────────────────────────────────
#  主对抗扰动生成器
# ─────────────────────────────────────────────

class AdversarialPerturbationGenerator:
    """
    对抗扰动生成器

    支持多种攻击策略：
    - PGD      (Projected Gradient Descent)     — 最强白盒基线
    - MI-FGSM  (Momentum Iterative FGSM)        — 动量加速，可迁移性更好
    - FGSM     (Fast Gradient Sign Method)      — 单步，速度最快

    使用 L∞ 范数约束确保扰动不可见性
    支持自适应 ε 动态调整（adaptive_epsilon=True）
    """

    def __init__(
        self,
        epsilon: float = 8 / 255,
        alpha: float = 1 / 255,
        num_steps: int = 40,
        momentum: float = 1.0,
        attack_type: str = "pgd",
        adaptive_epsilon: bool = True,
        adaptive_min_ratio: float = 0.5,
        adaptive_max_ratio: float = 1.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            epsilon:            L∞ 扰动上界基准值 (默认 8/255)
            alpha:              每步更新步长
            num_steps:          迭代次数
            momentum:           MI-FGSM 动量系数
            attack_type:        攻击类型 ['pgd', 'mifgsm', 'fgsm']
            adaptive_epsilon:   是否启用自适应扰动强度
            adaptive_min_ratio: 自适应 ε 最小倍率（相对于 epsilon_base）
            adaptive_max_ratio: 自适应 ε 最大倍率
            device:             计算设备
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.momentum = momentum
        self.attack_type = attack_type
        self.adaptive_epsilon = adaptive_epsilon
        self.device = device

        self.adaptive_scheduler = AdaptiveEpsilonScheduler(
            epsilon_base=epsilon,
            epsilon_min_ratio=adaptive_min_ratio,
            epsilon_max_ratio=adaptive_max_ratio,
        ) if adaptive_epsilon else None

    def generate(
        self,
        img: torch.Tensor,
        target_model: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        target_img: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成对抗扰动

        Args:
            img:          原始人脸图像 [B, C, H, W]，值域 [0, 1]
            target_model: 目标换脸模型（SimSwap 等）
            loss_fn:      自定义损失函数（可选）
            target_img:   目标人脸（可选，用于引导扰动方向）

        Returns:
            adv_img:      添加扰动后的图像 [B, C, H, W]
            perturbation: 扰动本身 [B, C, H, W]
        """
        img = img.to(self.device)
        target_model = target_model.to(self.device)
        target_model.eval()

        # 预计算自适应 ε 图（只算一次，全程复用）
        if self.adaptive_epsilon and self.adaptive_scheduler is not None:
            epsilon_map = self.adaptive_scheduler.compute_epsilon_map(img)
            epsilon_map = epsilon_map.expand(-1, 3, -1, -1)  # 扩展到 3 通道
        else:
            epsilon_map = None

        # 初始化扰动
        # PGD 用均匀随机初始化（标准做法）
        # FGSM / MI-FGSM 原本从零初始化，但零初始化导致第一步 adv_img==img，
        # 两个特征完全相同 → 余弦相似度=1 → 梯度=0 → sign(0)=0 → delta永远为零
        # 修复：统一用小随机噪声初始化打破对称性
        delta_data = torch.empty_like(img).uniform_(-self.epsilon, self.epsilon)
        if epsilon_map is not None:
            delta_data = torch.max(torch.min(delta_data, epsilon_map), -epsilon_map)

        delta = nn.Parameter(delta_data, requires_grad=True)

        momentum_grad = torch.zeros_like(img)

        for step in range(self.num_steps):
            adv_img = torch.clamp(img + delta, 0, 1)

            # 清零梯度
            if delta.grad is not None:
                delta.grad.zero_()

            feat_adv = target_model.get_id_feature(adv_img)
            loss = self._compute_loss(
                target_model, adv_img, img, feat_adv, target_img, loss_fn
            )
            loss.backward()

            grad = delta.grad.data.clone()

            # 计算更新方向
            if self.attack_type == "mifgsm":
                # 标准 MI-FGSM：L1归一化 → 动量累积 → 动量梯度的sign()
                grad = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
                momentum_grad = self.momentum * momentum_grad + grad
                update = self.alpha * momentum_grad.sign()
            else:
                update = self.alpha * grad.sign()

            # in-place 更新 delta.data，不创建新 tensor，保持 Parameter 身份
            with torch.no_grad():
                delta.data = delta.data + update
                # L∞ 投影：自适应 or 统一
                if epsilon_map is not None:
                    delta.data = self.adaptive_scheduler.project_with_map(delta.data, epsilon_map)
                else:
                    delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)

            if self.attack_type == "fgsm":
                break

        adv_img = torch.clamp(img + delta.detach(), 0, 1)
        perturbation = adv_img - img

        # 零扰动检测：如果生成的扰动全为零，打印警告便于排查
        max_pert = perturbation.abs().max().item()
        if max_pert < 1e-7:
            import warnings
            warnings.warn(
                f"[AdversarialPerturbationGenerator] 警告：{self.attack_type} 生成了零扰动！"
                f" epsilon={self.epsilon:.4f}, alpha={self.alpha:.4f}, steps={self.num_steps}"
                f" — 请检查 target_model.get_id_feature() 是否返回了常数输出或梯度为零。",
                RuntimeWarning, stacklevel=2
            )

        return adv_img, perturbation

    def _compute_loss(
        self,
        model: nn.Module,
        adv_img: torch.Tensor,
        orig_img: torch.Tensor,
        feat_adv: torch.Tensor,
        target_img: Optional[torch.Tensor],
        custom_loss_fn: Optional[nn.Module],
    ) -> torch.Tensor:
        """
        多目标损失函数
        L = -CosSim(f(x_adv), f(x))  +  λ * MSE(x_adv, x)

        - L_id:     最大化身份特征距离 → 欺骗换脸模型
        - L_percep: 最小化像素差异   → 保持视觉质量
        """
        if custom_loss_fn is not None:
            return custom_loss_fn(model, adv_img, orig_img, feat_adv, target_img)

        feat_orig = model.get_id_feature(orig_img).detach()
        l_id = -F.cosine_similarity(feat_adv, feat_orig).mean()
        l_percep = F.mse_loss(adv_img, orig_img)
        return l_id - 0.1 * l_percep

    def get_epsilon_map_visual(self, img: torch.Tensor) -> torch.Tensor:
        """
        返回自适应 ε 分布热力图（用于 UI 展示），值域 [0, 1]
        Returns: [B, 3, H, W]
        """
        if self.adaptive_scheduler is None:
            return torch.ones_like(img) * 0.5
        eps_map = self.adaptive_scheduler.compute_epsilon_map(img)
        eps_min = self.adaptive_scheduler.epsilon_min
        eps_max = self.adaptive_scheduler.epsilon_max
        vis = (eps_map - eps_min) / (eps_max - eps_min + 1e-8)
        return vis.expand(-1, 3, -1, -1)


# ─────────────────────────────────────────────
#  可配置多目标损失函数
# ─────────────────────────────────────────────

class MultiTargetLoss(nn.Module):
    """可配置多目标损失函数，适配消融实验的权重调整"""

    def __init__(
        self,
        lambda_id: float = 1.0,
        lambda_percep: float = 0.1,
        lambda_ssim: float = 0.05,
    ):
        super().__init__()
        self.lambda_id = lambda_id
        self.lambda_percep = lambda_percep
        self.lambda_ssim = lambda_ssim

    def forward(
        self,
        model: nn.Module,
        adv_img: torch.Tensor,
        orig_img: torch.Tensor,
        feat_adv: torch.Tensor,
        target_img: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat_orig = model.get_id_feature(orig_img).detach()
        l_id = -F.cosine_similarity(feat_adv, feat_orig).mean()
        l_percep = F.mse_loss(adv_img, orig_img)
        return self.lambda_id * l_id + self.lambda_percep * l_percep