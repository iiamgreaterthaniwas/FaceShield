"""
数据集下载与准备脚本

CelebA-HQ 获取方式（三选一）:
  方式A: 官方 Google Drive (需要翻墙)
  方式B: Kaggle 数据集
  方式C: 自行收集人脸图像 (最简单)

运行示例:
    # 使用自行收集的图像
    python scripts/prepare_dataset.py --mode custom --src_dir /path/to/your/faces

    # 用 Kaggle 下载 (需要 kaggle API key)
    python scripts/prepare_dataset.py --mode kaggle

    # 生成随机测试数据（仅用于调试代码）
    python scripts/prepare_dataset.py --mode dummy --num 50
"""

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
#  各种数据集准备方式
# ─────────────────────────────────────────────

def prepare_dummy_dataset(output_dir: Path, num_images: int = 50, img_size: int = 256):
    """
    生成随机假图像（仅用于调试代码流程，不用于实验）
    """
    output_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[数据集] 生成 {num_images} 张随机测试图像 (img_size={img_size})")
    for i in range(num_images):
        # 生成有层次感的随机人脸状图像
        arr = np.random.randint(100, 200, (img_size, img_size, 3), dtype=np.uint8)
        # 中心圆（模拟人脸区域）
        cx, cy = img_size // 2, img_size // 2
        for y in range(img_size):
            for x in range(img_size):
                if (x - cx) ** 2 + (y - cy) ** 2 < (img_size // 3) ** 2:
                    arr[y, x] = [200 + np.random.randint(-30, 30),
                                 160 + np.random.randint(-20, 20),
                                 130 + np.random.randint(-20, 20)]

        img = Image.fromarray(arr)
        img.save(output_dir / f"dummy_{i:04d}.png")

    print(f"[数据集] 已生成 {num_images} 张占位图像: {output_dir}")
    print("[注意] 这些是随机图像，仅供代码调试用。正式实验请使用 CelebA-HQ 数据集。")


def prepare_custom_dataset(src_dir: str, output_dir: Path, img_size: int = 256):
    """
    使用自行收集的人脸图像
    支持任意 jpg/png 格式
    """
    src_dir = Path(src_dir)
    output_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    img_paths = sorted([p for p in src_dir.rglob("*") if p.suffix.lower() in extensions])

    if not img_paths:
        print(f"[错误] 在 {src_dir} 中未找到图像文件")
        return

    print(f"[数据集] 发现 {len(img_paths)} 张源图像，开始预处理...")
    success = 0
    for i, path in enumerate(img_paths):
        try:
            img = Image.open(path).convert("RGB")
            # 中心裁剪后缩放（保留人脸主体）
            w, h = img.size
            min_side = min(w, h)
            left = (w - min_side) // 2
            top = (h - min_side) // 2
            img = img.crop((left, top, left + min_side, top + min_side))
            img = img.resize((img_size, img_size), Image.LANCZOS)
            img.save(output_dir / f"face_{i:04d}.png")
            success += 1
        except Exception as e:
            print(f"  跳过 {path.name}: {e}")

    print(f"[数据集] 完成: {success}/{len(img_paths)} 张已处理 → {output_dir}")


def prepare_kaggle_dataset(output_dir: Path, img_size: int = 256):
    """
    通过 Kaggle API 下载 CelebA-HQ 子集
    前提：已配置 ~/.kaggle/kaggle.json

    Kaggle 数据集：
    - jessicali9530/celeba-dataset (CelebA 原版 202k 张)
    - lamsimon/celebahq (CelebA-HQ 高清版)
    """
    print("[数据集] 尝试通过 Kaggle API 下载...")

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "lamsimon/celebahq",
             "--path", str(output_dir / "kaggle_download"), "--unzip"],
            check=True
        )
        print("[数据集] Kaggle 下载成功")

        # 找到解压后的图像
        downloaded = list((output_dir / "kaggle_download").rglob("*.jpg"))
        if not downloaded:
            downloaded = list((output_dir / "kaggle_download").rglob("*.png"))

        prepare_custom_dataset(
            str(output_dir / "kaggle_download"),
            output_dir,
            img_size,
        )

    except FileNotFoundError:
        print("[错误] 未安装 kaggle CLI，请运行: pip install kaggle")
        print("       并配置 API key: https://www.kaggle.com/docs/api")
    except subprocess.CalledProcessError as e:
        print(f"[错误] Kaggle 下载失败: {e}")
        print("       请检查 ~/.kaggle/kaggle.json 是否配置正确")


def check_dataset(data_dir: Path) -> dict:
    """检查数据集状态"""
    img_dir = data_dir / "images"
    if not img_dir.exists():
        img_dir = data_dir

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in extensions]

    stats = {
        "total_images": len(imgs),
        "path": str(img_dir),
        "ready": len(imgs) >= 10,
    }

    if imgs:
        sample = Image.open(imgs[0])
        stats["sample_size"] = f"{sample.size[0]}×{sample.size[1]}"

    return stats


# ─────────────────────────────────────────────
#  CelebA-HQ 下载指引
# ─────────────────────────────────────────────

DATASET_GUIDE = """
╔══════════════════════════════════════════════════════════╗
║          CelebA-HQ 数据集获取指引                         ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  方式 A: Kaggle (推荐)                                    ║
║  ─────────────────────────────────────────────────────  ║
║  1. 注册 Kaggle 账号: https://www.kaggle.com              ║
║  2. 创建 API Token，下载 kaggle.json 到 ~/.kaggle/        ║
║  3. pip install kaggle                                    ║
║  4. kaggle datasets download -d lamsimon/celebahq         ║
║                                                          ║
║  方式 B: 百度网盘 (国内用户)                               ║
║  ─────────────────────────────────────────────────────  ║
║  搜索 "CelebA-HQ 百度网盘" 或使用以下第三方镜像:           ║
║  https://github.com/tkarras/progressive_growing_of_gans  ║
║                                                          ║
║  方式 C: 自行收集 (最简单)                                 ║
║  ─────────────────────────────────────────────────────  ║
║  收集 100+ 张清晰人脸正面照，放入 data/raw/custom/        ║
║  然后运行:                                                ║
║    python scripts/prepare_dataset.py --mode custom       ║
║       --src_dir data/raw/custom                          ║
║                                                          ║
║  方式 D: 调试模式 (代码验证)                               ║
║  ─────────────────────────────────────────────────────  ║
║    python scripts/prepare_dataset.py --mode dummy        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(description="数据集准备工具")
    parser.add_argument("--mode", type=str, default="dummy",
                        choices=["dummy", "custom", "kaggle", "check"],
                        help="准备模式")
    parser.add_argument("--src_dir", type=str, default="data/raw/custom",
                        help="源图像目录 (--mode custom 时使用)")
    parser.add_argument("--output_dir", type=str, default="data/processed/celeba_hq")
    parser.add_argument("--num", type=int, default=50, help="dummy 模式生成数量")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--show_guide", action="store_true", help="显示下载指引")
    args = parser.parse_args()

    if args.show_guide:
        print(DATASET_GUIDE)
        return

    output_dir = Path(args.output_dir)

    if args.mode == "dummy":
        prepare_dummy_dataset(output_dir, args.num, args.img_size)

    elif args.mode == "custom":
        prepare_custom_dataset(args.src_dir, output_dir, args.img_size)

    elif args.mode == "kaggle":
        prepare_kaggle_dataset(output_dir, args.img_size)

    elif args.mode == "check":
        stats = check_dataset(output_dir)
        print(f"[检查] 数据集路径: {stats['path']}")
        print(f"[检查] 图像总数:   {stats['total_images']}")
        if "sample_size" in stats:
            print(f"[检查] 示例尺寸:   {stats['sample_size']}")
        print(f"[检查] 状态:       {'✅ 就绪' if stats['ready'] else '❌ 图像不足（< 10 张）'}")


if __name__ == "__main__":
    main()