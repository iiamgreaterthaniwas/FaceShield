"""
主实验脚本
完整的攻防实验流程：
1. 数据集加载
2. 对抗扰动生成（支持自适应 ε）
3. 换脸攻击测试
4. 指标评估
5. 结果可视化（折线图 + 热力图）

运行:
    # 单次实验
    python scripts/run_experiment.py --data_dir data/processed/celeba_hq \
        --epsilon 8 --steps 40 --attack pgd --num_samples 100

    # 对比多组参数（自动生成热力图）
    python scripts/run_experiment.py --compare_methods
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns  # 热力图用 seaborn

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.perturbation import AdversarialPerturbationGenerator, MultiTargetLoss
from models.simswap_wrapper import SimSwapWrapper
from utils.dataset import get_dataloader, save_image, tensor_to_numpy
from evaluation.metrics import Evaluator, compute_psnr, compute_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="人脸隐私保护系统 - 实验脚本")
    parser.add_argument("--data_dir",    type=str,   default="data/processed/celeba_hq")
    parser.add_argument("--output_dir",  type=str,   default="data/results")
    parser.add_argument("--model_path",  type=str,   default=None)
    parser.add_argument("--epsilon",     type=float, default=8.0)
    parser.add_argument("--steps",       type=int,   default=40)
    parser.add_argument("--attack",      type=str,   default="pgd",
                        choices=["pgd", "mifgsm", "fgsm"])
    parser.add_argument("--adaptive",    action="store_true", default=True,
                        help="启用自适应 ε 策略")
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--num_samples", type=int,   default=100)
    parser.add_argument("--img_size",    type=int,   default=256)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--compare_methods", action="store_true",
                        help="对比多组参数，生成折线图和热力图")
    return parser.parse_args()


# ─────────────────────────────────────────────
#  单组实验
# ─────────────────────────────────────────────

def run_single_experiment(
    model, dataloader, epsilon, num_steps, attack_type,
    output_dir, device, save_images=False, adaptive=True,
) -> dict:
    evaluator = Evaluator(device=device)
    generator = AdversarialPerturbationGenerator(
        epsilon=epsilon / 255.0,
        alpha=epsilon / 255.0 / 10,
        num_steps=num_steps,
        attack_type=attack_type,
        adaptive_epsilon=adaptive,
        device=device,
    )

    all_orig, all_adv = [], []
    batch_count = 0
    print(f"\n[实验] ε={epsilon}/255 | 算法={attack_type} | 步数={num_steps} | 自适应={adaptive}")
    print(f"{'Batch':>6} | {'PSNR':>8} | {'SSIM':>6} | {'ASR':>6} | {'时间':>6}")
    print("-" * 48)

    for batch in dataloader:
        imgs = batch["image"].to(device)

        t0 = time.time()
        with torch.enable_grad():
            adv_imgs, perturbations = generator.generate(imgs, model)
        elapsed = time.time() - t0

        target_imgs = torch.roll(imgs, shifts=1, dims=0)
        metrics = evaluator.evaluate_batch(
            model, imgs, adv_imgs, target_imgs, epsilon / 255.0
        )

        print(
            f"{batch_count:>6} | {metrics['psnr']:>8.2f} | "
            f"{metrics['ssim']:>6.4f} | {metrics['asr']:>6.2%} | {elapsed:.2f}s"
        )

        all_orig.append(imgs[:2].detach().cpu())
        all_adv.append(adv_imgs[:2].detach().cpu())

        if save_images and batch_count < 5:
            img_dir = output_dir / "sample_images"
            img_dir.mkdir(exist_ok=True)
            for i in range(min(2, imgs.shape[0])):
                save_image(imgs[i:i+1],     str(img_dir / f"b{batch_count}_i{i}_orig.png"))
                save_image(adv_imgs[i:i+1], str(img_dir / f"b{batch_count}_i{i}_adv.png"))

        batch_count += 1

    summary = evaluator.print_report()

    # 保存对比图
    orig_cat = torch.cat(all_orig[:3], dim=0)
    adv_cat  = torch.cat(all_adv[:3],  dim=0)
    save_comparison_grid(orig_cat, adv_cat, output_dir / f"comparison_eps{int(epsilon)}.png")

    return summary


# ─────────────────────────────────────────────
#  可视化函数
# ─────────────────────────────────────────────

def save_comparison_grid(orig, adv, save_path):
    """原图 vs 对抗图 vs 扰动 三行对比网格"""
    n = min(orig.shape[0], 4)
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))
    if n == 1:
        axes = axes.reshape(3, 1)

    for i in range(n):
        o = tensor_to_numpy(orig[i:i+1])
        a = tensor_to_numpy(adv[i:i+1])
        pert = np.clip(
            (a.astype(np.float32) - o.astype(np.float32)) * 10 + 128, 0, 255
        ).astype(np.uint8)

        axes[0, i].imshow(o);    axes[0, i].set_title("原始图像", fontsize=9);    axes[0, i].axis("off")
        axes[1, i].imshow(a);    axes[1, i].set_title("保护后图像", fontsize=9);  axes[1, i].axis("off")
        axes[2, i].imshow(pert); axes[2, i].set_title("扰动(×10)", fontsize=9);   axes[2, i].axis("off")

    plt.suptitle("人脸隐私保护效果对比", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 对比图已保存: {save_path}")


def plot_line_charts(results: dict, output_dir: Path):
    """折线图：不同 ε 下 PSNR 和 ASR 的变化趋势"""
    epsilons = sorted(results.keys())
    psnrs = [results[e].get("mean_psnr", 0) for e in epsilons]
    asrs  = [results[e].get("mean_asr",  0) * 100 for e in epsilons]
    ssims = [results[e].get("mean_ssim", 0) for e in epsilons]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # PSNR
    axes[0].plot(epsilons, psnrs, "b-o", linewidth=2, markersize=7)
    axes[0].axhline(y=30, color="r", linestyle="--", label="目标 30dB")
    axes[0].set_xlabel("ε (像素值/255)"); axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("扰动强度 vs 图像质量 (PSNR)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # ASR
    axes[1].plot(epsilons, asrs, "g-s", linewidth=2, markersize=7)
    axes[1].axhline(y=80, color="r", linestyle="--", label="目标 80%")
    axes[1].set_xlabel("ε (像素值/255)"); axes[1].set_ylabel("ASR (%)")
    axes[1].set_title("扰动强度 vs 防御成功率 (ASR)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # SSIM
    axes[2].plot(epsilons, ssims, "m-^", linewidth=2, markersize=7)
    axes[2].axhline(y=0.95, color="r", linestyle="--", label="目标 0.95")
    axes[2].set_xlabel("ε (像素值/255)"); axes[2].set_ylabel("SSIM")
    axes[2].set_title("扰动强度 vs 结构相似度 (SSIM)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle("消融实验：扰动强度 ε 对各指标的影响", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = output_dir / "line_epsilon_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 折线图已保存: {save_path}")


def plot_heatmaps(results_2d: dict, output_dir: Path):
    """
    热力图：二维参数扫描结果可视化
    X 轴：迭代次数 (steps)
    Y 轴：扰动强度 ε
    颜色：ASR / PSNR / SSIM

    Args:
        results_2d: {(epsilon, steps): {"mean_asr": ..., "mean_psnr": ..., "mean_ssim": ...}}
    """
    if not results_2d:
        return

    # 提取参数组合
    epsilons = sorted(set(k[0] for k in results_2d))
    steps_list = sorted(set(k[1] for k in results_2d))

    n_eps   = len(epsilons)
    n_steps = len(steps_list)

    # 构建二维矩阵
    asr_mat  = np.zeros((n_eps, n_steps))
    psnr_mat = np.zeros((n_eps, n_steps))
    ssim_mat = np.zeros((n_eps, n_steps))

    for i, eps in enumerate(epsilons):
        for j, steps in enumerate(steps_list):
            key = (eps, steps)
            if key in results_2d:
                asr_mat[i, j]  = results_2d[key].get("mean_asr",  0) * 100
                psnr_mat[i, j] = results_2d[key].get("mean_psnr", 0)
                ssim_mat[i, j] = results_2d[key].get("mean_ssim", 0)

    eps_labels   = [f"ε={e}" for e in epsilons]
    steps_labels = [f"T={s}" for s in steps_list]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── ASR 热力图 ──
    sns.heatmap(
        asr_mat,
        ax=axes[0],
        annot=True, fmt=".1f",
        xticklabels=steps_labels,
        yticklabels=eps_labels,
        cmap="YlOrRd",        # 暖色：越深防御越强
        vmin=0, vmax=100,
        linewidths=0.5,
        cbar_kws={"label": "ASR (%)"},
    )
    axes[0].set_title("防御成功率 ASR (%)\n（越深越好）", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("迭代次数 T"); axes[0].set_ylabel("扰动强度 ε")

    # ── PSNR 热力图 ──
    sns.heatmap(
        psnr_mat,
        ax=axes[1],
        annot=True, fmt=".1f",
        xticklabels=steps_labels,
        yticklabels=eps_labels,
        cmap="Blues_r",       # 蓝色反向：越浅 PSNR 越高（质量越好）
        linewidths=0.5,
        cbar_kws={"label": "PSNR (dB)"},
    )
    axes[1].set_title("图像质量 PSNR (dB)\n（越浅越好，目标>30）", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("迭代次数 T"); axes[1].set_ylabel("扰动强度 ε")

    # ── SSIM 热力图 ──
    sns.heatmap(
        ssim_mat,
        ax=axes[2],
        annot=True, fmt=".3f",
        xticklabels=steps_labels,
        yticklabels=eps_labels,
        cmap="Greens_r",      # 绿色反向：越浅 SSIM 越高
        vmin=0.8, vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "SSIM"},
    )
    axes[2].set_title("结构相似度 SSIM\n（越浅越好，目标>0.95）", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("迭代次数 T"); axes[2].set_ylabel("扰动强度 ε")

    plt.suptitle(
        "参数扫描热力图：ε × 迭代次数 对三项核心指标的影响",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_path = output_dir / "heatmap_param_scan.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 热力图已保存: {save_path}")


def plot_algorithm_comparison(algo_results: dict, output_dir: Path):
    """
    算法对比柱状图：FGSM vs PGD vs MI-FGSM
    对比三种算法在固定 ε 下的 ASR / PSNR / SSIM
    """
    algos = list(algo_results.keys())
    asrs  = [algo_results[a].get("mean_asr",  0) * 100 for a in algos]
    psnrs = [algo_results[a].get("mean_psnr", 0) for a in algos]
    ssims = [algo_results[a].get("mean_ssim", 0) for a in algos]

    x = np.arange(len(algos))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, vals, label, target, unit in zip(
        axes,
        [asrs, psnrs, ssims],
        ["ASR (%)", "PSNR (dB)", "SSIM"],
        [80, 30, 0.95],
        ["%", "dB", ""],
    ):
        bars = ax.bar(x, vals, color=colors, width=0.5, zorder=3)
        ax.axhline(y=target, color="r", linestyle="--", linewidth=1.2,
                   label=f"目标: {target}{unit}")
        ax.set_xticks(x); ax.set_xticklabels(algos, fontsize=11)
        ax.set_ylabel(label)
        ax.set_title(f"算法对比 — {label}")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, zorder=0)

        # 数值标注
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("三种攻击算法性能对比（ε=8/255）", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = output_dir / "algo_comparison_bar.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 算法对比柱状图已保存: {save_path}")


# ─────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[主程序] 运行设备: {device.upper()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SimSwapWrapper(model_path=args.model_path, img_size=args.img_size)
    model = model.to(device).eval()

    dataloader = get_dataloader(
        root=args.data_dir,
        split="test",
        img_size=args.img_size,
        batch_size=args.batch_size,
        max_samples=args.num_samples,
    )

    if args.compare_methods:
        # ── 二维参数扫描：生成热力图 ──
        epsilon_list = [2, 4, 8, 12, 16]
        steps_list   = [10, 20, 40, 60]
        results_1d   = {}   # {epsilon: summary} 用于折线图
        results_2d   = {}   # {(epsilon, steps): summary} 用于热力图
        algo_results = {}   # {algo: summary} 用于算法对比

        print("\n=== 参数扫描实验（生成热力图）===")
        for eps in epsilon_list:
            for steps in steps_list:
                summary = run_single_experiment(
                    model, dataloader, eps, steps,
                    args.attack, output_dir, device,
                    save_images=False, adaptive=args.adaptive,
                )
                results_2d[(eps, steps)] = summary
                # 收集标准步数 (40) 的结果用于折线图
                if steps == 40:
                    results_1d[eps] = summary

        print("\n=== 算法对比实验 ===")
        for algo in ["fgsm", "pgd", "mifgsm"]:
            summary = run_single_experiment(
                model, dataloader, 8, 40,
                algo, output_dir, device,
                save_images=False, adaptive=args.adaptive,
            )
            algo_results[algo.upper()] = summary

        # 生成所有图表
        plot_line_charts(results_1d, output_dir)
        plot_heatmaps(results_2d, output_dir)
        plot_algorithm_comparison(algo_results, output_dir)

    else:
        # 单次实验
        run_single_experiment(
            model, dataloader, args.epsilon, args.steps,
            args.attack, output_dir, device,
            save_images=args.save_images, adaptive=args.adaptive,
        )

    print(f"\n[完成] 所有结果已保存至 {output_dir}")


if __name__ == "__main__":
    main()