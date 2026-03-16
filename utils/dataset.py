"""
数据集准备与加载模块
支持 CelebA-HQ 数据集
提供数据预处理、增强和批量加载功能
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ─────────────────────────────────────────────
#  数据集类
# ─────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    人脸数据集加载器
    支持 CelebA-HQ 等标准格式

    目录结构：
        data/raw/
            celeba_hq/
                images/
                    000001.jpg
                    000002.jpg
                    ...
    """

    def __init__(
            self,
            root: str,
            split: str = "train",
            img_size: int = 256,
            max_samples: Optional[int] = None,
            augment: bool = False,
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment

        # 搜集图片路径
        self.img_paths = self._collect_images(split, max_samples)
        print(f"[Dataset] {split} 集共加载 {len(self.img_paths)} 张图像")

        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0,255] → [0,1]
        ])

        # 数据增强（训练时可选）
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def _collect_images(self, split: str, max_samples: Optional[int]) -> List[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        img_dir = self.root / "images"

        if not img_dir.exists():
            # 兼容直接放在 root 下的情况
            img_dir = self.root

        all_paths = sorted([
            p for p in img_dir.rglob("*")
            if p.suffix.lower() in extensions
        ])

        # 简单按比例划分 train/val/test
        n = len(all_paths)
        if split == "train":
            paths = all_paths[:int(n * 0.8)]
        elif split == "val":
            paths = all_paths[int(n * 0.8):int(n * 0.9)]
        else:  # test
            paths = all_paths[int(n * 0.9):]

        if max_samples:
            paths = paths[:max_samples]

        return paths

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.img_paths[idx]

        # 加载图像
        img = Image.open(img_path).convert("RGB")

        if self.augment:
            img = self.aug_transform(img)

        img_tensor = self.base_transform(img)

        return {
            "image": img_tensor,  # [3, H, W]
            "path": str(img_path),
            "idx": idx,
        }


class PairedFaceDataset(Dataset):
    """
    配对人脸数据集（源图 + 目标图）
    用于测试换脸+对抗扰动效果
    """

    def __init__(self, root: str, img_size: int = 256, max_pairs: int = 100):
        self.base = FaceDataset(root, split="test", img_size=img_size)
        self.max_pairs = min(max_pairs, len(self.base))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return self.max_pairs

    def __getitem__(self, idx: int) -> dict:
        # 随机选取不同的目标图
        target_idx = (idx + random.randint(1, len(self.base) - 1)) % len(self.base)

        src = self.base[idx]
        tgt = self.base[target_idx]

        return {
            "source": src["image"],
            "target": tgt["image"],
            "source_path": src["path"],
            "target_path": tgt["path"],
        }


# ─────────────────────────────────────────────
#  数据集准备脚本
# ─────────────────────────────────────────────

def prepare_celeba_hq(raw_dir: str, output_dir: str, img_size: int = 256):
    """
    CelebA-HQ 数据集预处理

    步骤：
    1. 统一图像尺寸为 img_size × img_size
    2. 转换为 RGB，保存为 PNG
    3. 过滤无效图像

    Args:
        raw_dir:    原始数据目录
        output_dir: 处理后输出目录
        img_size:   目标图像尺寸
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    img_paths = [p for p in raw_dir.rglob("*") if p.suffix.lower() in extensions]

    print(f"[准备数据集] 共发现 {len(img_paths)} 张原始图像")
    success, failed = 0, 0

    for path in img_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((img_size, img_size), Image.LANCZOS)
            out_path = output_dir / (path.stem + ".png")
            img.save(out_path)
            success += 1
        except Exception as e:
            print(f"  跳过 {path.name}: {e}")
            failed += 1

    print(f"[准备数据集] 完成: {success} 成功, {failed} 失败")
    print(f"[准备数据集] 输出目录: {output_dir}")


def collect_test_images(source_dir: str, output_dir: str, num_images: int = 100):
    """
    从来源目录收集测试图片（用于论文实验部分）
    可接受来自网络或 CelebA-HQ 的人脸图片
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png"}
    all_imgs = sorted([p for p in source_dir.rglob("*") if p.suffix.lower() in extensions])

    selected = all_imgs[:num_images]
    print(f"[测试集] 从 {len(all_imgs)} 张中选取 {len(selected)} 张")

    for i, src_path in enumerate(selected):
        dst = output_dir / f"test_{i:04d}{src_path.suffix}"
        import shutil
        shutil.copy2(src_path, dst)

    print(f"[测试集] 已复制到 {output_dir}")


# ─────────────────────────────────────────────
#  DataLoader 工厂函数
# ─────────────────────────────────────────────

def get_dataloader(
        root: str,
        split: str = "test",
        img_size: int = 256,
        batch_size: int = 4,
        num_workers: int = 2,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
) -> DataLoader:
    dataset = FaceDataset(root, split=split, img_size=img_size, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_paired_dataloader(
        root: str,
        img_size: int = 256,
        batch_size: int = 2,
        num_pairs: int = 100,
) -> DataLoader:
    dataset = PairedFaceDataset(root, img_size=img_size, max_pairs=num_pairs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# ─────────────────────────────────────────────
#  图像工具函数
# ─────────────────────────────────────────────

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """[B,C,H,W] tensor → [H,W,C] uint8 numpy，供 OpenCV/PIL 使用"""
    img = tensor.detach().cpu().squeeze(0)
    img = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return img


def numpy_to_tensor(img: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """[H,W,C] uint8 numpy → [1,C,H,W] float tensor"""
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def save_image(tensor: torch.Tensor, path: str):
    """保存 tensor 为图像文件"""
    img = tensor_to_numpy(tensor)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[保存] {path}")