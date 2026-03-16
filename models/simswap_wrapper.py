"""
SimSwap 模型封装模块
提供统一接口用于特征提取与换脸推理
支持作为白盒攻击目标模型
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class IDExtractor(nn.Module):
    """
    身份特征提取器
    使用预训练的人脸识别网络提取身份嵌入向量
    在实际项目中替换为 ArcFace / CosFace 等
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # 使用 ResNet50 作为 backbone（实际部署用 ArcFace）
        backbone = models.resnet50(pretrained=pretrained)
        # 去掉最后分类层，保留特征提取部分
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.embed = nn.Linear(2048, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 人脸图像 [B, 3, 224, 224]，值域 [0, 1]
        Returns:
            id_feat: 归一化身份嵌入 [B, 512]
        """
        feat = self.features(x).flatten(1)
        embed = self.embed(feat)
        return F.normalize(embed, dim=1)


class AttributeEncoder(nn.Module):
    """属性编码器：提取非身份属性（姿态、表情、背景等）"""

    def __init__(self):
        super().__init__()
        # 简化版属性编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SimSwapWrapper(nn.Module):
    """
    SimSwap 模型封装

    提供：
    1. get_id_feature()：提取身份特征（白盒攻击入口）
    2. swap_face()：执行换脸
    3. 支持梯度反向传播

    使用说明：
    - 实际部署时，请下载官方 SimSwap 预训练权重
    - 权重路径：checkpoints/simswap_512.pth
    - 参考：https://github.com/neuralchen/SimSwap
    """

    def __init__(self, model_path: Optional[str] = None, img_size: int = 224):
        super().__init__()
        self.img_size = img_size

        # 身份编码器（攻击目标）
        self.id_extractor = IDExtractor(pretrained=True)

        # 属性编码器
        self.attr_encoder = AttributeEncoder()

        # 生成器（简化 U-Net 结构）
        self.generator = self._build_generator()

        if model_path:
            self.load_pretrained(model_path)

    def _build_generator(self) -> nn.Module:
        """简化版换脸生成器（U-Net 结构）"""
        return nn.Sequential(
            # 编码器
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 解码器
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def get_id_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取身份特征向量（攻击的主要目标）
        梯度可以通过此函数回传到输入图像

        Args:
            x: [B, 3, H, W]
        Returns:
            id_feat: [B, 512]
        """
        x_resized = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.id_extractor(x_resized)

    def swap_face(
            self,
            source: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行换脸操作

        Args:
            source: 源人脸（提供身份）[B, 3, H, W]
            target: 目标人脸（提供属性）[B, 3, H, W]
        Returns:
            swapped: 换脸结果 [B, 3, H, W]
        """
        id_feat = self.get_id_feature(source)
        attr_feat = self.attr_encoder(target)
        # 简化版融合（实际 SimSwap 使用 AdaIN 调制）
        swapped = self.generator(target)
        return (swapped + 1) / 2  # Tanh [-1,1] → [0,1]

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.swap_face(source, target)

    def load_pretrained(self, path: str):
        """加载预训练权重"""
        try:
            # latest_net_G.pth 就在 path 目录下
            pth_file = os.path.join(path, "latest_net_G.pth")
            state = torch.load(pth_file, map_location="cpu")
            self.generator.load_state_dict(state, strict=False)
            print(f"[SimSwap] 已加载预训练权重: {pth_file}")
        except FileNotFoundError:
            print(f"[SimSwap] 警告: 权重文件不存在 {path}")
        except Exception as e:
            print(f"[SimSwap] 加载失败: {e}")