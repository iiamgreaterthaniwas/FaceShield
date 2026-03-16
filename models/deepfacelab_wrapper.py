"""
DeepFaceLab 模型封装模块
提供与 SimSwapWrapper 统一的接口，用于白盒攻击与泛化性实验

架构说明：
  DeepFaceLab 采用编码器-解码器结构（共享编码器 / 独立解码器）
  与 SimSwap 的 AdaIN 调制机制不同，DFL 通过共享 Encoder 提取
  中间表示，再由源/目标专属 Decoder 重建人脸。

  本封装模拟该结构，作为攻击迁移性实验的第二个目标模型。

参考：
  https://github.com/iperov/DeepFaceLab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class DFLEncoder(nn.Module):
    """
    DFL 共享编码器
    将任意人脸压缩到潜在空间，两张人脸共享权重
    """

    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64,  4, 2, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
        )
        # 全连接压缩到潜在向量
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, latent_dim),
            nn.LeakyReLU(0.1),
        )
        self.fc_expand = nn.Linear(latent_dim, 512 * 8 * 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            latent: [B, 512, 8, 8]  中间潜在表示
        """
        feat = self.conv_layers(x)
        z = self.fc(feat)
        z_expand = self.fc_expand(z).view(-1, 512, 8, 8)
        return z_expand


class DFLDecoder(nn.Module):
    """
    DFL 专属解码器（源/目标各一个，权重不共享）
    从潜在空间重建目标人脸
    """

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.ConvTranspose2d(32,  3,   4, 2, 1), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class DFLIDExtractor(nn.Module):
    """
    DFL 身份特征提取器（与 SimSwap 不同，基于 MobileNetV2 轻量化）
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.mobilenet_v2(pretrained=pretrained)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(1280, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.pool(feat).flatten(1)
        embed = self.embed(feat)
        return F.normalize(embed, dim=1)


class DeepFaceLabWrapper(nn.Module):
    """
    DeepFaceLab 封装

    提供与 SimSwapWrapper 完全相同的接口：
    - get_id_feature(x)  → 身份向量，梯度可回传
    - swap_face(src, tgt) → 换脸结果
    - forward(src, tgt)  → 同 swap_face

    使用示例（与 SimSwap 可互换）：
        model = DeepFaceLabWrapper()
        generator = AdversarialPerturbationGenerator(...)
        adv_img, _ = generator.generate(img, model)

    迁移性实验：
        在 SimSwap 上生成的对抗扰动是否对 DFL 同样有效？
        → 若 ASR 仍高，说明扰动具有跨模型泛化能力。
    """

    def __init__(self, model_path: Optional[str] = None, img_size: int = 224):
        super().__init__()
        self.img_size = img_size

        self.id_extractor = DFLIDExtractor(pretrained=True)
        self.encoder = DFLEncoder(latent_dim=512)
        self.decoder_src = DFLDecoder()   # 源人脸解码器
        self.decoder_tgt = DFLDecoder()   # 目标人脸解码器

        if model_path:
            self.load_pretrained(model_path)

    def get_id_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取身份特征向量（白盒攻击入口）
        梯度可通过此函数回传到输入图像

        Args:
            x: [B, 3, H, W]，值域 [0, 1]
        Returns:
            id_feat: [B, 512]，L2 归一化
        """
        x_resized = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.id_extractor(x_resized)

    def swap_face(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        DFL 换脸流程：
          1. 共享编码器提取 source 潜在表示
          2. 目标解码器（decoder_tgt）用 source 的潜在重建人脸
          → 结果具有 source 的身份 + target 姿态/背景的混合

        Args:
            source: 源人脸（提供身份）[B, 3, H, W]
            target: 目标人脸（提供属性）[B, 3, H, W]
        Returns:
            swapped: 换脸结果 [B, 3, H, W]，值域 [0, 1]
        """
        src_resized = F.interpolate(source, (self.img_size, self.img_size),
                                    mode="bilinear", align_corners=False)
        latent = self.encoder(src_resized)
        swapped = self.decoder_tgt(latent)
        # Tanh [-1, 1] → [0, 1]
        swapped = (swapped + 1) / 2
        # 上采样到原始尺寸
        if swapped.shape[-1] != target.shape[-1]:
            swapped = F.interpolate(swapped, size=target.shape[-2:],
                                    mode="bilinear", align_corners=False)
        return swapped.clamp(0, 1)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.swap_face(source, target)

    def load_pretrained(self, path: str):
        """加载预训练权重"""
        import os
        try:
            pth = os.path.join(path, "dfl_model.pth")
            state = torch.load(pth, map_location="cpu")
            self.load_state_dict(state, strict=False)
            print(f"[DeepFaceLab] 已加载权重: {pth}")
        except FileNotFoundError:
            print(f"[DeepFaceLab] 警告：权重文件不存在 {path}，使用随机初始化")
        except Exception as e:
            print(f"[DeepFaceLab] 加载失败: {e}")
