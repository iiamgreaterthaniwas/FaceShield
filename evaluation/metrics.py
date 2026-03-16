"""
评估指标模块
实现论文中的三个核心评估维度：
1. ASR  (Attack Success Rate) - 防御成功率
2. PSNR (Peak Signal-to-Noise Ratio) - 图像质量
3. SSIM (Structural Similarity Index) - 结构相似度
以及处理时间统计
"""

import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


# ─────────────────────────────────────────────
#  图像质量指标
# ─────────────────────────────────────────────

def compute_psnr(orig: torch.Tensor, adv: torch.Tensor) -> float:
    """
    计算峰值信噪比 PSNR
    越高说明图像质量保持越好（无感知变化），目标 > 30 dB

    Args:
        orig, adv: [B, C, H, W] float tensor，值域 [0, 1]
    Returns:
        PSNR 值（dB）；若扰动为零（两图完全相同）返回 100.0 作为上界占位值
    """
    orig_np = orig.detach().cpu().numpy()
    adv_np  = adv.detach().cpu().numpy()

    # 若扰动为零则直接返回上界，避免除零 RuntimeWarning
    if np.allclose(orig_np, adv_np, atol=1e-7):
        return 100.0

    psnr_list = []
    for i in range(orig_np.shape[0]):
        o   = np.transpose(orig_np[i], (1, 2, 0))  # CHW → HWC
        a   = np.transpose(adv_np[i],  (1, 2, 0))
        val = psnr_fn(o, a, data_range=1.0)
        # 单张也可能因浮点误差产生 inf，用上界替换
        psnr_list.append(val if np.isfinite(val) else 100.0)

    return float(np.mean(psnr_list))


def compute_ssim(orig: torch.Tensor, adv: torch.Tensor) -> float:
    """
    计算结构相似度 SSIM
    越高说明人眼感知的结构保持越好，目标 > 0.95

    Args:
        orig, adv: [B, C, H, W] float tensor，值域 [0, 1]
    Returns:
        SSIM 值 [0, 1]
    """
    orig_np = orig.detach().cpu().numpy()
    adv_np = adv.detach().cpu().numpy()

    ssim_list = []
    for i in range(orig_np.shape[0]):
        o = np.transpose(orig_np[i], (1, 2, 0))
        a = np.transpose(adv_np[i], (1, 2, 0))
        s = ssim_fn(o, a, channel_axis=2, data_range=1.0)
        ssim_list.append(s)

    return float(np.mean(ssim_list))


def compute_l_inf(orig: torch.Tensor, adv: torch.Tensor) -> float:
    """计算扰动的 L∞ 范数（验证约束是否生效）"""
    diff = (adv - orig).abs().max().item()
    return diff


# ─────────────────────────────────────────────
#  防御成功率 (ASR)
# ─────────────────────────────────────────────

def compute_asr(
        swap_model,
        orig_source: torch.Tensor,
        target: torch.Tensor,
        adv_source: torch.Tensor,
        threshold: float = 0.5,
        device: str = "cuda",
) -> Dict[str, float]:
    """
    计算攻击成功率 (Attack Success Rate)

    判定标准：
    - 对比 [正常换脸] vs [带扰动换脸] 结果的身份相似度
    - 若相似度低于阈值，认为换脸被干扰成功

    Args:
        swap_model:    换脸模型
        orig_source:   原始源人脸（无扰动）[B, 3, H, W]
        target:        目标人脸 [B, 3, H, W]
        adv_source:    加扰动后的源人脸 [B, 3, H, W]
        threshold:     余弦相似度判断阈值

    Returns:
        {
            "asr": 防御成功率（越高越好）,
            "avg_cosine_normal": 正常换脸平均身份相似度,
            "avg_cosine_adv": 扰动换脸平均身份相似度,
        }
    """
    swap_model = swap_model.to(device).eval()
    orig_source = orig_source.to(device)
    target = target.to(device)
    adv_source = adv_source.to(device)

    with torch.no_grad():
        # 正常换脸结果的身份特征
        normal_result = swap_model.swap_face(orig_source, target)
        feat_normal = swap_model.get_id_feature(normal_result)
        feat_orig = swap_model.get_id_feature(orig_source)
        cosine_normal = F.cosine_similarity(feat_normal, feat_orig).mean().item()

        # 带扰动换脸结果的身份特征
        adv_result = swap_model.swap_face(adv_source, target)
        feat_adv_result = swap_model.get_id_feature(adv_result)
        feat_adv_src = swap_model.get_id_feature(adv_source)
        cosine_adv = F.cosine_similarity(feat_adv_result, feat_adv_src).mean().item()

        # 防御成功：扰动后换脸结果与原始身份相似度 < 阈值
        batch_success = (F.cosine_similarity(feat_adv_result, feat_orig) < threshold).float()
        asr = batch_success.mean().item()

    return {
        "asr": asr,
        "avg_cosine_normal": cosine_normal,
        "avg_cosine_adv": cosine_adv,
        "identity_drop": cosine_normal - cosine_adv,
    }


# ─────────────────────────────────────────────
#  综合评估器
# ─────────────────────────────────────────────

class Evaluator:
    """批量评估器，汇总所有指标并生成报告"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results: List[Dict] = []

    def evaluate_batch(
            self,
            swap_model,
            orig_imgs: torch.Tensor,
            adv_imgs: torch.Tensor,
            target_imgs: torch.Tensor,
            epsilon: float,
    ) -> Dict[str, float]:
        """
        对一批图像计算全部指标

        Returns:
            包含 psnr, ssim, l_inf, asr 等指标的字典
        """
        # 图像质量指标
        psnr = compute_psnr(orig_imgs, adv_imgs)
        ssim = compute_ssim(orig_imgs, adv_imgs)
        l_inf = compute_l_inf(orig_imgs, adv_imgs)

        # 防御成功率
        asr_metrics = compute_asr(
            swap_model, orig_imgs, target_imgs, adv_imgs,
            threshold=0.5, device=self.device
        )

        result = {
            "psnr": psnr,
            "ssim": ssim,
            "l_inf": l_inf,
            "epsilon_budget": epsilon,
            **asr_metrics,
        }
        self.results.append(result)
        return result

    def time_inference(self, perturbation_fn, img: torch.Tensor, n_runs: int = 5) -> float:
        """测量单张图片处理时间（秒）"""
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            perturbation_fn(img)
            times.append(time.time() - t0)
        return float(np.mean(times[1:]))  # 去掉首次预热

    def summary(self) -> Dict[str, float]:
        """汇总所有批次的平均指标"""
        if not self.results:
            return {}

        keys = self.results[0].keys()
        summary = {}
        for k in keys:
            vals = [r[k] for r in self.results if isinstance(r[k], (int, float))]
            summary[f"mean_{k}"] = float(np.mean(vals))
            summary[f"std_{k}"] = float(np.std(vals))

        return summary

    def print_report(self):
        """打印评估报告"""
        s = self.summary()
        print("\n" + "=" * 50)
        print("    评估报告 (人脸隐私保护系统)")
        print("=" * 50)
        print(f"  防御成功率 (ASR):     {s.get('mean_asr', 0):.2%}")
        print(f"  图像质量 (PSNR):      {s.get('mean_psnr', 0):.2f} dB")
        print(f"  结构相似度 (SSIM):    {s.get('mean_ssim', 0):.4f}")
        print(f"  L∞ 扰动:             {s.get('mean_l_inf', 0):.4f}")
        print(f"  身份相似度下降:       {s.get('mean_identity_drop', 0):.4f}")
        print("=" * 50)
        return s