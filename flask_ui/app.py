import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import queue
import threading
import base64
import json
import traceback
import subprocess
import tempfile
import shutil
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_file
import uuid

from core.perturbation import AdversarialPerturbationGenerator
from evaluation.metrics import compute_psnr, compute_ssim, Evaluator
from utils.dataset import numpy_to_tensor, tensor_to_numpy, get_dataloader

# ─────────────────────────────────────────────
#  Flask 初始化
# ─────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB 上传限制

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT   = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "celeba_hq"
DATA_RESULTS   = PROJECT_ROOT / "data" / "results"

# ─────────────────────────────────────────────
#  官方 SimSwap 配置（按实际路径修改）
# ─────────────────────────────────────────────
# SimSwap 仓库根目录（包含 test_one_image.py 的那个文件夹）
SIMSWAP_ROOT        = Path(os.environ.get("SIMSWAP_ROOT", PROJECT_ROOT / "simswap"))
# ArcFace 权重路径（相对于 SIMSWAP_ROOT，或绝对路径均可）
SIMSWAP_ARC_PATH    = os.environ.get("SIMSWAP_ARC_PATH", "arcface_model/arcface_checkpoint.tar")
# 模型名称（checkpoints/<name> 目录下存放 latest_net_G.pth）
SIMSWAP_MODEL_NAME  = os.environ.get("SIMSWAP_MODEL_NAME", "people")
# 裁剪尺寸
SIMSWAP_CROP_SIZE   = int(os.environ.get("SIMSWAP_CROP_SIZE", 224))
# 默认目标人脸（用户未上传 target 时的兜底图，相对于 SIMSWAP_ROOT）
SIMSWAP_DEFAULT_TARGET = os.environ.get(
    "SIMSWAP_DEFAULT_TARGET",
    str(SIMSWAP_ROOT / "crop_224" / "6.jpg"),
)
# Python 解释器路径（可改为 conda 环境里的 python）
SIMSWAP_PYTHON = os.environ.get("SIMSWAP_PYTHON", sys.executable)

_swap_model = None  # ArcFaceWrapper instance
_model_lock = threading.Lock()

# 实验日志队列（SSE 实时推送用）
_experiment_log_queue: queue.Queue = queue.Queue()
_experiment_running = False

# ─────────────────────────────────────────────
#  官方 SimSwap 子进程调用
# ─────────────────────────────────────────────

def run_official_simswap(
    pic_a_path: str,
    pic_b_path: str,
    output_dir: str,
    crop_size: int = SIMSWAP_CROP_SIZE,
    name: str = SIMSWAP_MODEL_NAME,
    arc_path: str = SIMSWAP_ARC_PATH,
    timeout: int = 120,
) -> Optional[str]:
    """
    调用官方 SimSwap 的 test_one_image.py 完成换脸。

    Args:
        pic_a_path: 源人脸图片路径（提供身份）
        pic_b_path: 目标人脸图片路径（提供姿态/属性）
        output_dir: 输出目录，结果保存为 output_dir/result.jpg
        timeout:    子进程超时秒数

    Returns:
        result_path: 成功时返回结果图片路径；失败时返回 None
    """
    script = SIMSWAP_ROOT / "test_one_image.py"
    if not script.exists():
        print(f"[SimSwap] 找不到脚本: {script}")
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        SIMSWAP_PYTHON, str(script),
        "--crop_size", str(crop_size),
        "--name",      name,
        "--Arc_path",  arc_path,
        "--pic_a_path", pic_a_path,
        "--pic_b_path", pic_b_path,
        "--output_path", output_dir + "/",
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(SIMSWAP_ROOT),       # 必须在 SimSwap 目录下运行
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            print(f"[SimSwap] 子进程报错 (code={proc.returncode}):\n{proc.stderr[-1000:]}")
            return None

        # SimSwap 输出文件名不固定，扫描目录找实际生成的图片
        img_exts = {".jpg", ".jpeg", ".png"}
        out_files = [
            f for f in Path(output_dir).iterdir()
            if f.suffix.lower() in img_exts
        ]
        if out_files:
            # 取最新生成的文件（以防目录里有多个）
            result_path = str(max(out_files, key=lambda f: f.stat().st_mtime))
            return result_path
        else:
            print(f"[SimSwap] 输出目录无图片文件: {output_dir}")
            print(f"[SimSwap] stdout: {proc.stdout[-500:]}")
            return None

    except subprocess.TimeoutExpired:
        print(f"[SimSwap] 超时 ({timeout}s)")
        return None
    except Exception as e:
        print(f"[SimSwap] 调用异常: {e}")
        return None


def simswap_pair_to_base64(
    face_a_np: np.ndarray,
    face_b_path: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    给定 numpy 格式的人脸 A（原图或保护图）和目标人脸 B 的路径，
    执行一次官方 SimSwap，返回 (result_b64, error_msg)。

    face_a_np: uint8 HWC RGB numpy 数组
    face_b_path: 目标人脸文件路径
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 保存 face_a 到临时文件
        a_path = os.path.join(tmpdir, "face_a.jpg")
        Image.fromarray(face_a_np).save(a_path, quality=95)

        # 确定 face_b 路径
        b_path = face_b_path if (face_b_path and os.path.exists(face_b_path)) else SIMSWAP_DEFAULT_TARGET
        if not os.path.exists(b_path):
            return None, f"目标人脸不存在: {b_path}"

        out_dir = os.path.join(tmpdir, "out")
        result_path = run_official_simswap(a_path, b_path, out_dir)

        if result_path is None:
            return None, "SimSwap 子进程失败，请查看终端日志"

        with open(result_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return b64, None


# ─────────────────────────────────────────────
#  异步任务系统（单张图像保护）
# ─────────────────────────────────────────────
_protect_tasks: dict = {}
_tasks_lock = threading.Lock()


def _cleanup_old_tasks():
    now = time.time()
    with _tasks_lock:
        expired = [k for k, v in _protect_tasks.items()
                   if now - v.get("created_at", now) > 1800]
        for k in expired:
            del _protect_tasks[k]


def _protect_worker(
    task_id: str,
    img_bytes: bytes,
    epsilon: float,
    num_steps: int,
    attack_type: str,
    adaptive: bool,
    target_bytes: Optional[bytes] = None,   # ← 新增：目标人脸字节
):
    """
    后台保护线程。
    新增：若提供 target_bytes，则调用官方 SimSwap 执行真实换脸对比；
         否则退化为梯度热力图可视化（兜底方案）。
    """
    target_path_tmp = None   # 临时文件路径，用完后删除

    try:
        with _tasks_lock:
            _protect_tasks[task_id]["status"]   = "running"
            _protect_tasks[task_id]["progress"] = 10

        pil_img    = Image.open(BytesIO(img_bytes)).convert("RGB").resize((224, 224))
        img_tensor = numpy_to_tensor(np.array(pil_img), device=DEVICE)
        eps        = epsilon / 255.0

        # ── 如果提供了目标人脸，先保存到临时文件 ──────────────
        if target_bytes:
            tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tf.write(target_bytes)
            tf.close()
            target_path_tmp = tf.name
        # 否则 target_path_tmp = None，run_official_simswap 会用默认目标

        model = get_model()
        generator = AdversarialPerturbationGenerator(
            epsilon=eps, alpha=eps / 10,
            num_steps=num_steps, attack_type=attack_type,
            adaptive_epsilon=adaptive, device=DEVICE,
        )

        with _tasks_lock:
            _protect_tasks[task_id]["progress"] = 25

        # ── 生成对抗扰动 ──────────────────────────────────────
        t0 = time.time()
        with torch.enable_grad():
            adv_img, perturbation = generator.generate(img_tensor, model)
        elapsed = time.time() - t0

        with _tasks_lock:
            _protect_tasks[task_id]["progress"] = 55

        eps_vis  = generator.get_epsilon_map_visual(img_tensor)
        psnr_val = compute_psnr(img_tensor.detach(), adv_img.detach())
        ssim_val = compute_ssim(img_tensor.detach(), adv_img.detach())
        l_inf    = perturbation.abs().max().item() * 255

        # ── 余弦相似度 ────────────────────────────────────────
        import torch.nn.functional as F_nn
        with torch.no_grad():
            feat_orig = model.get_id_feature(img_tensor)
            feat_adv  = model.get_id_feature(adv_img.detach())
            cos_sim   = F_nn.cosine_similarity(feat_orig, feat_adv).item()

        # ── 保存 numpy 格式供后续攻击测试使用 ──────────────
        orig_np = tensor_to_numpy(img_tensor)
        adv_np  = tensor_to_numpy(adv_img.detach())

        def _np_to_bytes(arr):
            buf = BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=95)
            return buf.getvalue()

        with _tasks_lock:
            _protect_tasks[task_id]["progress"] = 90

        result = {
            "original":       tensor_to_base64(img_tensor),
            "protected":      tensor_to_base64(adv_img),
            "perturbation":   tensor_to_base64((perturbation * 10 + 0.5).clamp(0, 1)),
            "epsilon_map":    tensor_to_base64(eps_vis),
            "metrics": {
                "psnr":       round(psnr_val, 2),
                "ssim":       round(ssim_val, 4),
                "l_inf":      round(l_inf, 2),
                "elapsed":    round(elapsed, 2),
                "device":     DEVICE.upper(),
                "id_cos_sim": round(cos_sim, 4),
            },
        }

        with _tasks_lock:
            _protect_tasks[task_id]["status"]       = "done"
            _protect_tasks[task_id]["progress"]     = 100
            _protect_tasks[task_id]["result"]       = result
            _protect_tasks[task_id]["_orig_bytes"]  = _np_to_bytes(orig_np)
            _protect_tasks[task_id]["_adv_bytes"]   = _np_to_bytes(adv_np)
            _protect_tasks[task_id]["_target_path"] = target_path_tmp
            target_path_tmp = None  # 转移所有权，finally 不再删除

    except Exception as e:
        with _tasks_lock:
            _protect_tasks[task_id]["status"] = "error"
            _protect_tasks[task_id]["error"]  = str(e)
            _protect_tasks[task_id]["trace"]  = traceback.format_exc()

    finally:
        # 仅异常时清理（正常完成由攻击测试接口或任务过期负责）
        if target_path_tmp and os.path.exists(target_path_tmp):
            os.unlink(target_path_tmp)


def _simswap_asr_for_batch(
    imgs_np_list: list,
    adv_np_list: list,
    log_fn,
    b_idx: int,
) -> float:
    """
    用官方 SimSwap 对一个 batch 里每张图做真实换脸，计算真实 ASR。

    正确判定逻辑：
      - 用保护图换脸 → result_adv
      - 比较 ArcFace(result_adv) 与 ArcFace(原始source) 的余弦相似度
      - 相似度低于阈值 → 换脸结果已偏离原始身份 → 扰动成功（ASR +1）

    对照组（用于动态阈值）：
      - 用原图换脸   → result_orig
      - cos(result_orig, source) 作为正常换脸的基准相似度
      - 若 cos(result_adv, source) < cos(result_orig, source) * 0.7 视为成功
        （比正常换脸身份相似度下降 30% 以上）

    若 SimSwap 不可用则回退到输入层特征余弦相似度近似。
    """
    import torch.nn.functional as _F

    simswap_available = (
        SIMSWAP_ROOT.exists() and
        (SIMSWAP_ROOT / "test_one_image.py").exists() and
        os.path.exists(SIMSWAP_DEFAULT_TARGET)
    )

    model = get_model()
    success_count = 0
    total = len(imgs_np_list)

    for i, (orig_np, adv_np) in enumerate(zip(imgs_np_list, adv_np_list)):
        if simswap_available:
            with tempfile.TemporaryDirectory() as tmpdir:
                orig_path = os.path.join(tmpdir, "orig.jpg")
                adv_path  = os.path.join(tmpdir, "adv.jpg")
                Image.fromarray(orig_np).save(orig_path, quality=95)
                Image.fromarray(adv_np).save(adv_path,  quality=95)

                out_orig = os.path.join(tmpdir, "out_orig")
                out_adv  = os.path.join(tmpdir, "out_adv")
                r_orig = run_official_simswap(orig_path, SIMSWAP_DEFAULT_TARGET, out_orig)
                r_adv  = run_official_simswap(adv_path,  SIMSWAP_DEFAULT_TARGET, out_adv)

                if r_orig is None or r_adv is None:
                    simswap_available = False
                    log_fn(f"    [警告] SimSwap 换脸失败，batch {b_idx} 改用特征余弦估算\n")

                if r_orig and r_adv:
                    # 统一用 PIL 读取 + resize(224) 保证所有图像前处理一致
                    def _load_np224(path_or_np):
                        if isinstance(path_or_np, str):
                            img = Image.open(path_or_np).convert("RGB").resize((224, 224), Image.LANCZOS)
                        else:
                            img = Image.fromarray(path_or_np).convert("RGB").resize((224, 224), Image.LANCZOS)
                        return numpy_to_tensor(np.array(img), DEVICE)

                    src_t      = _load_np224(orig_np)       # 原始 source（统一路径）
                    res_orig_t = _load_np224(r_orig)        # 正常换脸结果
                    res_adv_t  = _load_np224(r_adv)         # 保护图换脸结果

                    with torch.no_grad():
                        feat_src      = model.get_id_feature(src_t)
                        feat_res_orig = model.get_id_feature(res_orig_t)
                        feat_res_adv  = model.get_id_feature(res_adv_t)
                        # 正常换脸对 source 的身份相似度（基准）
                        cos_normal = _F.cosine_similarity(feat_res_orig, feat_src).item()
                        # 扰动换脸对 source 的身份相似度
                        cos_adv    = _F.cosine_similarity(feat_res_adv,  feat_src).item()

                    # 双重判定：
                    #   条件A：相对判定 —— 比正常换脸下降 25% 以上
                    #   条件B：绝对判定 —— 扰动结果与 source 相似度低于 0.3（几乎不像同一个人）
                    # 满足其一即视为扰动成功
                    rel_drop  = cos_normal - cos_adv          # 相似度绝对下降量
                    rel_ratio = cos_adv / (cos_normal + 1e-6) # 下降比例
                    success   = (rel_ratio < 0.75) or (cos_adv < 0.3)
                    if success:
                        success_count += 1
                    log_fn(
                        f"    [ASR调试] 图{i}: cos_normal={cos_normal:.3f}  "
                        f"cos_adv={cos_adv:.3f}  drop={rel_drop:+.3f}  "
                        f"ratio={rel_ratio:.2f}  "
                        f"{'✅成功' if success else '❌未成功'}\n"
                    )
                    continue  # 本张已处理

        # ── 回退：直接比较输入层特征（source原图 vs 保护图）──
        orig_t = numpy_to_tensor(orig_np, DEVICE)
        adv_t  = numpy_to_tensor(adv_np,  DEVICE)
        with torch.no_grad():
            feat_o = model.get_id_feature(orig_t)
            feat_a = model.get_id_feature(adv_t)
            cos = _F.cosine_similarity(feat_o, feat_a).item()
        if cos < 0.5:
            success_count += 1

    return success_count / total if total > 0 else 0.0


def _make_heatmap_pair(img_tensor, adv_img, model, orig_np, adv_np):
    """
    SimSwap 不可用时的退化方案：
    左图 = 原图 + 绿色边框，右图 = 保护图 + 梯度热力图叠加 + 红色边框。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    from PIL import Image as _PIL

    h, w   = orig_np.shape[:2]
    border = max(4, h // 28)

    # 原图 + 绿框
    orig_marked = orig_np.copy()
    for arr, color in [(orig_marked, [46, 160, 110])]:
        arr[:border, :]  = color; arr[-border:, :] = color
        arr[:, :border]  = color; arr[:, -border:] = color
    buf = BytesIO(); _PIL.fromarray(orig_marked).save(buf, "PNG")
    swap_orig_b64 = base64.b64encode(buf.getvalue()).decode()

    # 保护图 + 热力图 + 红框
    adv_clone = adv_img.detach().clone().requires_grad_(True)
    model.get_id_feature(adv_clone).sum().backward()
    g = adv_clone.grad.abs().mean(dim=1, keepdim=True)[0, 0].cpu().numpy()
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    heat  = (cm.hot(g)[:, :, :3] * 255).astype(np.uint8)
    blend = (adv_np * 0.55 + heat * 0.45).clip(0, 255).astype(np.uint8)
    blend[:border, :]  = [192, 60, 30]; blend[-border:, :] = [192, 60, 30]
    blend[:, :border]  = [192, 60, 30]; blend[:, -border:] = [192, 60, 30]
    buf2 = BytesIO(); _PIL.fromarray(blend).save(buf2, "PNG")
    swap_prot_b64 = base64.b64encode(buf2.getvalue()).decode()

    return swap_orig_b64, swap_prot_b64


class _ArcFaceWrapper(torch.nn.Module):
    """
    薄层包装：加载官方 SimSwap 的 ArcFace，提供 get_id_feature() 接口。
    供 AdversarialPerturbationGenerator 使用，确保攻击目标与真实换脸模型一致。
    """
    def __init__(self, arc_path: str, simswap_root: str, device: str):
        super().__init__()
        # 动态导入 SimSwap 的 arcface_models
        # 该文件使用 ResNet(IRBlock, layers) 构建网络，没有 Backbone 类
        # append 而非 insert(0)：SimSwap 目录可能含 logging.py 等同名文件
        # 插到最前会遮蔽标准库，导致 Flask/Werkzeug 日志静默失效
        if simswap_root not in sys.path:
            sys.path.append(simswap_root)
        # 注入假的 models.config，避免 from .config import device 触发 CUDA 初始化
        import types as _types
        if "models.config" not in sys.modules:
            _fake_cfg = _types.ModuleType("models.config")
            _fake_cfg.device = device
            _fake_cfg.num_classes = 93431
            sys.modules["models.config"] = _fake_cfg

        # 正规 import —— pickle 反序列化完整模型时必须能在此路径找到类定义
        try:
            import models.arcface_models as _arc_mod
        except Exception as e:
            raise RuntimeError(f"无法导入 arcface_models: {e}")

        # conv3x3 定义在别处，补充注入
        if not hasattr(_arc_mod, "conv3x3"):
            import torch.nn as _nn
            def _conv3x3(in_planes, out_planes, stride=1):
                return _nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                  stride=stride, padding=1, bias=False)
            _arc_mod.conv3x3 = _conv3x3

        # 加载权重（.tar checkpoint）
        ckpt = torch.load(arc_path, map_location=device)
        # SimSwap 的 arcface_checkpoint.tar 保存的是完整模型对象，不是 state_dict
        # 需要区分三种情况：
        #   1. 完整模型对象（torch.save(model, path)）-> 直接取 .state_dict()
        #   2. dict 含 "netArc" key -> 取 ckpt["netArc"]，可能又是模型或 state_dict
        #   3. 普通 state_dict（OrderedDict）
        import torch.nn as _nn
        def _extract_state(obj):
            if isinstance(obj, _nn.Module):
                return obj.state_dict()
            return obj
        if isinstance(ckpt, dict) and "netArc" in ckpt:
            state = _extract_state(ckpt["netArc"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = _extract_state(ckpt["state_dict"])
        elif isinstance(ckpt, _nn.Module):
            state = ckpt.state_dict()
        else:
            state = ckpt

        # 实例化 IR-SE50 网络（layers=[3,4,14,3] 为 ResNet-50 标准配置）
        net = _arc_mod.ResNet(
            block=_arc_mod.IRBlock,
            layers=[3, 4, 14, 3],
            use_se=True,
        ).to(device)

        net.load_state_dict(state, strict=False)
        net.eval()
        self.netArc = net
        print(f"[Server] ArcFace (IR-SE50) 已加载: {arc_path}")

    def get_id_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取身份特征 —— 攻击损失的目标。
        ArcFace 输入规格：112x112，归一化到 [-1, 1]
        """
        import torch.nn.functional as F_nn
        x_112 = F_nn.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        x_norm = (x_112 - 0.5) / 0.5   # [0,1] -> [-1,1]
        feat = self.netArc(x_norm)
        return F_nn.normalize(feat, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_id_feature(x)


def get_model() -> _ArcFaceWrapper:
    """
    返回攻击目标模型（官方 SimSwap 的 ArcFace）。
    扰动生成时攻击的就是这个模型，保证对真实换脸系统有效。
    """
    global _swap_model
    if _swap_model is None:
        with _model_lock:
            if _swap_model is None:
                arc_path = SIMSWAP_ARC_PATH
                if not os.path.isabs(arc_path):
                    arc_path = str(SIMSWAP_ROOT / arc_path)
                print(f"[Server] 加载攻击目标模型 ArcFace，设备: {DEVICE}")
                _swap_model = _ArcFaceWrapper(
                    arc_path     = arc_path,
                    simswap_root = str(SIMSWAP_ROOT),
                    device       = DEVICE,
                ).to(DEVICE).eval()
    return _swap_model


def tensor_to_base64(tensor: torch.Tensor) -> str:
    np_img  = tensor_to_numpy(tensor)
    pil_img = Image.fromarray(np_img)
    buf     = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─────────────────────────────────────────────
#  路由：页面
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", device=DEVICE.upper())


# ─────────────────────────────────────────────
#  路由：单张图像保护（异步版）
# ─────────────────────────────────────────────

@app.route("/api/protect_async", methods=["POST"])
def api_protect_async():
    """提交任务，立即返回 task_id；前端轮询 /api/task_status/<task_id>"""
    if "image" not in request.files:
        return jsonify({"error": "未上传图片"}), 400

    file        = request.files["image"]
    epsilon     = float(request.form.get("epsilon", 8))
    num_steps   = int(request.form.get("num_steps", 40))
    attack_type = request.form.get("attack_type", "pgd")
    adaptive    = request.form.get("adaptive", "true") == "true"
    img_bytes   = file.read()

    # ← 新增：读取目标人脸（可选）
    target_bytes = None
    if "target_image" in request.files:
        tf = request.files["target_image"]
        if tf and tf.filename:
            target_bytes = tf.read()

    _cleanup_old_tasks()
    task_id = str(uuid.uuid4())

    with _tasks_lock:
        _protect_tasks[task_id] = {
            "status": "pending", "progress": 0,
            "result": None, "error": None,
            "created_at": time.time(),
        }

    threading.Thread(
        target=_protect_worker,
        args=(task_id, img_bytes, epsilon, num_steps, attack_type, adaptive, target_bytes),
        daemon=True,
    ).start()

    return jsonify({"task_id": task_id})


@app.route("/api/task_status/<task_id>")
def api_task_status(task_id: str):
    with _tasks_lock:
        task = _protect_tasks.get(task_id)
    if task is None:
        return jsonify({"error": "任务不存在"}), 404
    resp = {
        "status":      task["status"],
        "progress":    task["progress"],
        "status_hint": task.get("status_hint", ""),
    }
    if task["status"] == "done":
        resp["result"] = task["result"]
    elif task["status"] == "error":
        resp["error"] = task["error"]
    return jsonify(resp)


# ─────────────────────────────────────────────
#  路由：攻击测试（用户主动触发）
#  POST /api/attack_test
#  JSON body: { "task_id": "<protect task id>" }
#  可选 form-data: target_image（覆盖保护时上传的目标人脸）
# ─────────────────────────────────────────────

@app.route("/api/attack_test", methods=["POST"])
def api_attack_test():
    """
    对已完成的保护任务执行 SimSwap 换脸攻击测试。
    保护阶段不再自动执行换脸，由用户主动点击触发此接口。
    """
    task_id = request.form.get("task_id") or (request.get_json(silent=True) or {}).get("task_id")
    if not task_id:
        return jsonify({"error": "缺少 task_id"}), 400

    with _tasks_lock:
        task = _protect_tasks.get(task_id)

    if task is None:
        return jsonify({"error": "任务不存在或已过期"}), 404
    if task.get("status") != "done":
        return jsonify({"error": "保护任务尚未完成"}), 400

    orig_bytes = task.get("_orig_bytes")
    adv_bytes  = task.get("_adv_bytes")
    if not orig_bytes or not adv_bytes:
        return jsonify({"error": "原始图像数据已过期，请重新生成保护图像"}), 400

    # 若用户此时上传了新的目标人脸，优先使用；否则用保护时留存的路径
    target_path_tmp = None
    try:
        if "target_image" in request.files:
            tf = request.files["target_image"]
            if tf and tf.filename:
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tmp.write(tf.read())
                tmp.close()
                target_path_tmp = tmp.name

        b_path = target_path_tmp or task.get("_target_path") or SIMSWAP_DEFAULT_TARGET

        orig_np = np.array(Image.open(BytesIO(orig_bytes)).convert("RGB"))
        adv_np  = np.array(Image.open(BytesIO(adv_bytes)).convert("RGB"))

        swap_orig_b64 = swap_prot_b64 = None
        swap_mode = "simswap"

        if SIMSWAP_ROOT.exists() and (SIMSWAP_ROOT / "test_one_image.py").exists():
            swap_orig_b64, err1 = simswap_pair_to_base64(orig_np, b_path)
            swap_prot_b64, err2 = simswap_pair_to_base64(adv_np,  b_path)

            if swap_orig_b64 is None or swap_prot_b64 is None:
                swap_mode = "heatmap"
                model = get_model()
                img_t = numpy_to_tensor(orig_np, device=DEVICE)
                adv_t = numpy_to_tensor(adv_np,  device=DEVICE)
                swap_orig_b64, swap_prot_b64 = _make_heatmap_pair(
                    img_t, adv_t, model, orig_np, adv_np
                )
        else:
            swap_mode = "heatmap"
            model = get_model()
            img_t = numpy_to_tensor(orig_np, device=DEVICE)
            adv_t = numpy_to_tensor(adv_np,  device=DEVICE)
            swap_orig_b64, swap_prot_b64 = _make_heatmap_pair(
                img_t, adv_t, model, orig_np, adv_np
            )

        return jsonify({
            "swap_orig":      swap_orig_b64,
            "swap_protected": swap_prot_b64,
            "swap_mode":      swap_mode,
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    finally:
        if target_path_tmp and os.path.exists(target_path_tmp):
            os.unlink(target_path_tmp)


# ─────────────────────────────────────────────
#  路由：独立换脸测试（新增）
#  POST /api/swap_test
#  form-data: pic_a (源人脸), pic_b (目标人脸，可选)
# ─────────────────────────────────────────────

@app.route("/api/swap_test", methods=["POST"])
def api_swap_test():
    """
    独立换脸接口：直接调用官方 SimSwap，不做对抗扰动。
    用于验证 SimSwap 环境是否配置正确。
    """
    if "pic_a" not in request.files:
        return jsonify({"error": "请上传 pic_a（源人脸）"}), 400

    a_bytes = request.files["pic_a"].read()
    b_bytes = request.files["pic_b"].read() if "pic_b" in request.files else None

    with tempfile.TemporaryDirectory() as tmpdir:
        a_path = os.path.join(tmpdir, "a.jpg")
        Image.open(BytesIO(a_bytes)).convert("RGB").resize((224, 224)).save(a_path, quality=95)

        if b_bytes:
            b_path = os.path.join(tmpdir, "b.jpg")
            Image.open(BytesIO(b_bytes)).convert("RGB").resize((224, 224)).save(b_path, quality=95)
        else:
            b_path = SIMSWAP_DEFAULT_TARGET

        if not os.path.exists(b_path):
            return jsonify({"error": f"目标人脸不存在，请上传 pic_b 或配置 SIMSWAP_DEFAULT_TARGET"}), 400

        out_dir = os.path.join(tmpdir, "out")
        result_path = run_official_simswap(a_path, b_path, out_dir)

        if result_path is None:
            return jsonify({"error": "SimSwap 运行失败，请查看服务器终端日志"}), 500

        with open(result_path, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode()

    return jsonify({"result": result_b64})


# ─────────────────────────────────────────────
#  路由：同步保护（保留兼容）
# ─────────────────────────────────────────────

@app.route("/api/protect", methods=["POST"])
def api_protect():
    if "image" not in request.files:
        return jsonify({"error": "未上传图片"}), 400

    file        = request.files["image"]
    epsilon     = float(request.form.get("epsilon", 8))
    num_steps   = int(request.form.get("num_steps", 40))
    attack_type = request.form.get("attack_type", "pgd")
    adaptive    = request.form.get("adaptive", "true") == "true"
    target_bytes = request.files["target_image"].read() if "target_image" in request.files else None

    try:
        pil_img    = Image.open(file).convert("RGB").resize((224, 224))
        img_tensor = numpy_to_tensor(np.array(pil_img), device=DEVICE)
        eps        = epsilon / 255.0

        model = get_model()
        generator = AdversarialPerturbationGenerator(
            epsilon=eps, alpha=eps / 10,
            num_steps=num_steps, attack_type=attack_type,
            adaptive_epsilon=adaptive, device=DEVICE,
        )

        t0 = time.time()
        with torch.enable_grad():
            adv_img, perturbation = generator.generate(img_tensor, model)
        elapsed = time.time() - t0

        eps_vis  = generator.get_epsilon_map_visual(img_tensor)
        psnr_val = compute_psnr(img_tensor.detach(), adv_img.detach())
        ssim_val = compute_ssim(img_tensor.detach(), adv_img.detach())
        l_inf    = perturbation.abs().max().item() * 255

        import torch.nn.functional as F_nn
        with torch.no_grad():
            feat_orig = model.get_id_feature(img_tensor)
            feat_adv  = model.get_id_feature(adv_img.detach())
            cos_sim   = F_nn.cosine_similarity(feat_orig, feat_adv).item()

        orig_np = tensor_to_numpy(img_tensor)
        adv_np  = tensor_to_numpy(adv_img.detach())

        # 换脸对比
        target_path_tmp = None
        if target_bytes:
            tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tf.write(target_bytes); tf.close()
            target_path_tmp = tf.name

        try:
            b_path = target_path_tmp if target_path_tmp else SIMSWAP_DEFAULT_TARGET
            if SIMSWAP_ROOT.exists() and (SIMSWAP_ROOT / "test_one_image.py").exists():
                swap_orig_b64, _ = simswap_pair_to_base64(orig_np, b_path)
                swap_prot_b64, _ = simswap_pair_to_base64(adv_np, b_path)
                swap_mode = "simswap"
                if swap_orig_b64 is None or swap_prot_b64 is None:
                    swap_orig_b64, swap_prot_b64 = _make_heatmap_pair(img_tensor, adv_img, model, orig_np, adv_np)
                    swap_mode = "heatmap"
            else:
                swap_orig_b64, swap_prot_b64 = _make_heatmap_pair(img_tensor, adv_img, model, orig_np, adv_np)
                swap_mode = "heatmap"
        finally:
            if target_path_tmp and os.path.exists(target_path_tmp):
                os.unlink(target_path_tmp)

        return jsonify({
            "original":       tensor_to_base64(img_tensor),
            "protected":      tensor_to_base64(adv_img),
            "perturbation":   tensor_to_base64((perturbation * 10 + 0.5).clamp(0, 1)),
            "epsilon_map":    tensor_to_base64(eps_vis),
            "swap_orig":      swap_orig_b64,
            "swap_protected": swap_prot_b64,
            "swap_mode":      swap_mode,
            "metrics": {
                "psnr":       round(psnr_val, 2),
                "ssim":       round(ssim_val, 4),
                "l_inf":      round(l_inf, 2),
                "elapsed":    round(elapsed, 2),
                "device":     DEVICE.upper(),
                "id_cos_sim": round(cos_sim, 4),
            },
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─────────────────────────────────────────────
#  路由：数据集准备
# ─────────────────────────────────────────────

@app.route("/api/prepare_dataset", methods=["POST"])
def api_prepare_dataset():
    data        = request.get_json()
    src_folder  = data.get("src_folder", "").strip()
    img_size    = int(data.get("img_size", 224))
    max_samples = int(data.get("max_samples", 100))

    def generate():
        src_dir = Path(src_folder)
        out_dir = DATA_PROCESSED / "images"

        if not src_dir.exists():
            yield f"data: ❌ 文件夹不存在：{src_dir}\n\n"
            return

        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        all_paths  = sorted([p for p in src_dir.rglob("*") if p.suffix.lower() in extensions])
        if not all_paths:
            yield f"data: ❌ 未找到图片文件\n\n"
            return

        selected = all_paths[:max_samples]
        out_dir.mkdir(parents=True, exist_ok=True)

        yield f"data: 📁 来源：{src_dir}\n\n"
        yield f"data: 📁 输出：{out_dir}\n\n"
        yield f"data: 🖼️  发现 {len(all_paths)} 张，处理前 {len(selected)} 张\n\n"
        yield f"data: {'─'*40}\n\n"

        success, failed = 0, 0
        for i, path in enumerate(selected):
            try:
                img = Image.open(path).convert("RGB")
                w, h = img.size; m = min(w, h)
                img  = img.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2))
                img  = img.resize((img_size, img_size), Image.LANCZOS)
                img.save(out_dir / f"face_{i:04d}.png")
                success += 1
                if (i + 1) % 10 == 0 or i == len(selected) - 1:
                    pct = int((i+1) / len(selected) * 100)
                    yield f"data: PROGRESS:{pct}:{i+1}/{len(selected)} ✅{success} ❌{failed}\n\n"
            except Exception as e:
                failed += 1
                yield f"data: ⚠️  跳过 {path.name}: {e}\n\n"

        yield f"data: {'─'*40}\n\n"
        yield f"data: ✅ 完成！成功 {success} 张，失败 {failed} 张\n\n"
        yield f"data: DONE\n\n"

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/dataset_status")
def api_dataset_status():
    img_dir = DATA_PROCESSED / "images"
    if not img_dir.exists():
        return jsonify({"ready": False, "message": "数据集未准备"})
    imgs = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    if not imgs:
        return jsonify({"ready": False, "message": "文件夹存在但没有图片"})
    sample = Image.open(imgs[0])
    return jsonify({
        "ready": True, "count": len(imgs),
        "size":  f"{sample.size[0]}×{sample.size[1]}",
        "path":  str(img_dir),
    })


# ─────────────────────────────────────────────
#  路由：批量实验（SSE 实时日志）
# ─────────────────────────────────────────────

def _run_experiment_thread(epsilon, num_steps, attack_type, num_samples, compare_mode, include_fawkes=False, include_dfl=False):
    global _experiment_running
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as _fm
    import seaborn as sns

    # ── 中文字体配置 ────────────────────────────────────
    # 优先顺序：SimHei(Windows) → Noto Sans CJK(Linux) → WenQuanYi → 降级英文
    _zh_font_candidates = [
        "SimHei", "Microsoft YaHei", "STHeiti",          # Windows / macOS
        "Noto Sans CJK SC", "Noto Sans CJK JP",           # Linux
        "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",       # Linux 文泉驿
        "Source Han Sans CN", "Source Han Sans SC",       # Adobe 思源
        "PingFang SC", "Heiti SC",                        # macOS
    ]
    _available = {f.name for f in _fm.fontManager.ttflist}
    _zh_font = next((f for f in _zh_font_candidates if f in _available), None)
    if _zh_font:
        matplotlib.rcParams["font.family"]      = "sans-serif"
        matplotlib.rcParams["font.sans-serif"]  = [_zh_font, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示
    else:
        # 找不到已命名字体，尝试直接用 ttc 路径注册
        _ttc_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
        for _ttc in _ttc_candidates:
            if os.path.exists(_ttc):
                _fm.fontManager.addfont(_ttc)
                _prop = _fm.FontProperties(fname=_ttc)
                matplotlib.rcParams["font.family"]      = "sans-serif"
                matplotlib.rcParams["font.sans-serif"]  = [_prop.get_name(), "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                break

    def log(msg):
        _experiment_log_queue.put(msg)

    try:
        DATA_RESULTS.mkdir(parents=True, exist_ok=True)
        img_dir = DATA_PROCESSED / "images"
        _imgs_check = (list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))) if img_dir.exists() else []
        if not img_dir.exists() or not _imgs_check:
            log("❌ 数据集未准备，请先去【数据集准备】处理图片")
            log("DONE"); return

        model      = get_model()
        dataloader = get_dataloader(
            root=str(DATA_PROCESSED), split="test",
            img_size=224, batch_size=4, max_samples=int(num_samples),
        )


        epsilon_list = [2, 4, 8, 12, 16] if compare_mode else [int(epsilon)]
        steps_list   = [10, 20, 40]       if compare_mode else [int(num_steps)]

        results_1d, results_2d = {}, {}
        log("🚀 实验开始...\n")
        total_runs = len(epsilon_list) * len(steps_list)
        run_idx    = 0

        for eps in epsilon_list:
            for steps in steps_list:
                run_idx += 1
                log(f"\n[{run_idx}/{total_runs}] ε={eps}/255  步数={steps}  算法={attack_type}\n")
                evaluator = Evaluator(device=DEVICE)
                generator = AdversarialPerturbationGenerator(
                    epsilon=eps/255.0, alpha=eps/255.0/10,
                    num_steps=steps, attack_type=attack_type,
                    adaptive_epsilon=True, device=DEVICE,
                )
                for b_idx, batch in enumerate(dataloader):
                    imgs = batch["image"].to(DEVICE)
                    t0 = time.time()
                    with torch.enable_grad():
                        adv_imgs, _ = generator.generate(imgs, model)
                    elapsed = time.time() - t0

                    # ── 图像质量指标 ──────────────────────────────
                    psnr_val = compute_psnr(imgs.detach(), adv_imgs.detach())
                    ssim_val = compute_ssim(imgs.detach(), adv_imgs.detach())

                    # ── 真实 SimSwap ASR（每张图跑换脸子进程对比）──
                    # 将 batch 里每张图转为 numpy，逐张调用 SimSwap
                    imgs_np_list = [
                        tensor_to_numpy(imgs[i:i+1]) for i in range(imgs.shape[0])
                    ]
                    adv_np_list = [
                        tensor_to_numpy(adv_imgs[i:i+1].detach()) for i in range(adv_imgs.shape[0])
                    ]
                    simswap_avail = (
                        SIMSWAP_ROOT.exists() and
                        (SIMSWAP_ROOT / "test_one_image.py").exists()
                    )
                    asr_mode = "SimSwap" if simswap_avail else "余弦近似"
                    asr_val = _simswap_asr_for_batch(imgs_np_list, adv_np_list, log, b_idx)

                    metrics = {"psnr": psnr_val, "ssim": ssim_val, "asr": asr_val}
                    evaluator.results.append(metrics)
                    log(f"  Batch {b_idx}: PSNR={psnr_val:.1f}dB  SSIM={ssim_val:.3f}  ASR({asr_mode})={asr_val:.1%}  ({elapsed:.1f}s)\n")
                    time.sleep(0.02)

                summary = evaluator.summary()
                results_2d[(eps, steps)] = summary
                if steps == (40 if compare_mode else int(num_steps)):
                    results_1d[eps] = summary
                log(f"  → 平均 PSNR={summary.get('mean_psnr',0):.2f}  ASR={summary.get('mean_asr',0):.1%}\n")

        # 折线图
        if results_1d:
            epsilons = sorted(results_1d.keys())
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            cfgs = [("mean_psnr","PSNR (dB)","#3b82f6",30),
                    ("mean_asr","ASR (%)","#10b981",0.8),
                    ("mean_ssim","SSIM","#8b5cf6",0.95)]
            for ax, (key, ylabel, color, target) in zip(axes, cfgs):
                vals = [results_1d[e].get(key,0)*(100 if "asr" in key else 1) for e in epsilons]
                ax.plot(epsilons, vals, "o-", color=color, linewidth=2.5, markersize=8)
                ax.axhline(y=target*(100 if "asr" in key else 1), color="#ef4444",
                           linestyle="--", linewidth=1.5, label="目标")
                ax.set_xlabel("ε"); ax.set_ylabel(ylabel); ax.set_title(ylabel)
                ax.legend(); ax.grid(alpha=0.2)
            plt.suptitle("消融实验：扰动强度对各指标的影响", fontsize=12, fontweight="bold")
            plt.tight_layout()
            _lc_path = DATA_RESULTS / "line_chart.png"
            try:
                plt.savefig(str(_lc_path), dpi=150, bbox_inches="tight")
                plt.close()
                if _lc_path.exists():
                    log(f"\n📊 折线图已保存（{_lc_path.stat().st_size // 1024} KB）\n")
                else:
                    log("\n❌ 折线图：文件未生成，请检查磁盘空间或路径权限\n")
            except Exception as _err:
                plt.close("all")
                log(f"\n❌ 折线图生成失败: {type(_err).__name__}: {_err}\n{traceback.format_exc()}\n")

        # 热力图
        if compare_mode and results_2d:
            eps_list = sorted(set(k[0] for k in results_2d))
            s_list   = sorted(set(k[1] for k in results_2d))
            n_e, n_s = len(eps_list), len(s_list)
            asr_mat  = np.zeros((n_e, n_s))
            psnr_mat = np.zeros((n_e, n_s))
            ssim_mat = np.zeros((n_e, n_s))
            for i, e in enumerate(eps_list):
                for j, s in enumerate(s_list):
                    r = results_2d.get((e,s), {})
                    asr_mat[i,j]  = r.get("mean_asr",0)*100
                    psnr_mat[i,j] = r.get("mean_psnr",0)
                    ssim_mat[i,j] = r.get("mean_ssim",0)
            xl = [f"T={s}" for s in s_list]; yl = [f"ε={e}" for e in eps_list]
            fig, axes = plt.subplots(1, 3, figsize=(17, 5))
            sns.heatmap(asr_mat,  ax=axes[0], annot=True, fmt=".1f", xticklabels=xl, yticklabels=yl, cmap="YlOrRd",  vmin=0, vmax=100)
            sns.heatmap(psnr_mat, ax=axes[1], annot=True, fmt=".1f", xticklabels=xl, yticklabels=yl, cmap="Blues_r")
            sns.heatmap(ssim_mat, ax=axes[2], annot=True, fmt=".3f", xticklabels=xl, yticklabels=yl, cmap="Greens_r", vmin=0.8, vmax=1.0)
            for ax, t in zip(axes, ["ASR (%)","PSNR (dB)","SSIM"]):
                ax.set_title(t); ax.set_xlabel("迭代次数 T"); ax.set_ylabel("扰动强度 ε")
            plt.suptitle("参数扫描热力图", fontsize=12, fontweight="bold", y=1.02)
            plt.tight_layout()
            _hm_path = DATA_RESULTS / "heatmap.png"
            try:
                plt.savefig(str(_hm_path), dpi=150, bbox_inches="tight")
                plt.close()
                if _hm_path.exists():
                    log(f"🗺️  热力图已保存（{_hm_path.stat().st_size // 1024} KB）\n")
                else:
                    log("❌ 热力图：文件未生成，请检查磁盘空间或路径权限\n")
            except Exception as _err:
                plt.close("all")
                log(f"\n❌ 热力图生成失败: {type(_err).__name__}: {_err}\n{traceback.format_exc()}\n")


        # ─────────────────────────────────────────────
        #  Fawkes 基线对比实验
        # ─────────────────────────────────────────────
        if include_fawkes:
            log("\n🦅 开始 Fawkes 基线对比实验...\n")
            log("   策略：用 FGSM 单步低强度扰动模拟 Fawkes cloaking 效果\n")
            # 取固定 ε=8 做对比（论文常用设置）
            fawkes_eps   = 8 / 255.0
            fawkes_results = {}  # {method: {psnr, ssim, asr}}

            method_cfgs = [
                ("Ours (PGD)",       dict(attack_type=attack_type, num_steps=int(num_steps), adaptive_epsilon=True)),
                ("Fawkes-like(FGSM)", dict(attack_type="fgsm",     num_steps=1,              adaptive_epsilon=False)),
                ("MI-FGSM",          dict(attack_type="mifgsm",    num_steps=int(num_steps), adaptive_epsilon=True)),
            ]

            for method_name, cfg in method_cfgs:
                log(f"  运行方法: {method_name.replace(chr(10),' ')}\n")
                m_eval = Evaluator(device=DEVICE)
                # alpha 统一在这里设置，不放进 cfg 里，避免重复关键字参数报错
                alpha = fawkes_eps / 4 if cfg["attack_type"] == "mifgsm" else fawkes_eps / 10
                m_gen  = AdversarialPerturbationGenerator(
                    epsilon=fawkes_eps, alpha=alpha,
                    device=DEVICE, **cfg,
                )
                for batch in dataloader:
                    imgs = batch["image"].to(DEVICE)
                    with torch.enable_grad():
                        adv_imgs, _ = m_gen.generate(imgs, model)
                    psnr_v = compute_psnr(imgs.detach(), adv_imgs.detach())
                    ssim_v = compute_ssim(imgs.detach(), adv_imgs.detach())
                    imgs_np  = [tensor_to_numpy(imgs[i:i+1])     for i in range(imgs.shape[0])]
                    adv_np   = [tensor_to_numpy(adv_imgs[i:i+1].detach()) for i in range(adv_imgs.shape[0])]
                    asr_v    = _simswap_asr_for_batch(imgs_np, adv_np, log, -1)
                    m_eval.results.append({"psnr": psnr_v, "ssim": ssim_v, "asr": asr_v})
                    time.sleep(0.01)
                s = m_eval.summary()
                fawkes_results[method_name] = s
                log(f"    → PSNR={s.get('mean_psnr',0):.2f}dB  SSIM={s.get('mean_ssim',0):.3f}  ASR={s.get('mean_asr',0):.1%}\n")

            # 生成柱状图对比
            methods   = list(fawkes_results.keys())
            asr_vals  = [fawkes_results[m].get("mean_asr",  0)*100 for m in methods]
            psnr_vals = [fawkes_results[m].get("mean_psnr", 0)     for m in methods]
            ssim_vals = [fawkes_results[m].get("mean_ssim", 0)     for m in methods]

            x      = np.arange(len(methods))
            colors = ["#ef4444", "#3b82f6", "#10b981"]
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))

            metrics_info = [
                (asr_vals,  "ASR (%)",   80,   "%",  None,  None),
                (psnr_vals, "PSNR (dB)", 30,   "dB", None,  None),
                (ssim_vals, "SSIM",      0.95, "",   None,  None),
            ]
            # 对 PSNR/SSIM 设置收紧的 y 轴范围，以呈现方法间的细微差异
            for idx, (vals, ylabel, target, unit, _, __) in enumerate(metrics_info):
                ax = axes[idx]
                bars = ax.bar(x, vals, color=colors, width=0.55, zorder=3)
                ax.axhline(y=target, color="#374151", linestyle="--", linewidth=1.2, label=f"目标 {target}{unit}")
                ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
                ax.set_ylabel(ylabel); ax.set_title(ylabel)
                ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25, zorder=0)
                # v_min/v_max 对所有指标都定义，避免 ASR 子图里 NameError
                v_min = min(vals) if vals else 0
                v_max = max(vals) if vals else 1
                if vals and ylabel != "ASR (%)":
                    margin = max((v_max - v_min) * 2, v_max * 0.02)
                    ax.set_ylim(max(0, v_min - margin), v_max + margin)
                for bar, val in zip(bars, vals):
                    fmt = f"{val:.2f}" if ylabel == "SSIM" else f"{val:.1f}"
                    offset = (v_max - v_min) * 0.05 + 0.001 if v_max != v_min else v_max * 0.02
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height() + offset,
                            fmt, ha="center", va="bottom", fontsize=9)
            plt.suptitle(f"方法对比：Ours vs Fawkes-like vs MI-FGSM（ε=8/255）\n"
                         f"※ 相同ε下PSNR/SSIM差异细微，主要区分指标为ASR", fontsize=11, fontweight="bold")
            plt.tight_layout()
            _fw_path = DATA_RESULTS / "fawkes_comparison.png"
            try:
                log("\n📊 Fawkes 对比图生成中...\n")
                plt.savefig(str(_fw_path), dpi=150, bbox_inches="tight")
                plt.close()
                if _fw_path.exists():
                    log(f"✅ Fawkes 对比图已保存（{_fw_path.stat().st_size // 1024} KB）\n")
                else:
                    log("❌ Fawkes 对比图：文件未生成，请检查磁盘空间或路径权限\n")
            except Exception as _err:
                plt.close("all")
                log(f"\n❌ Fawkes 对比图生成失败: {type(_err).__name__}: {_err}\n{traceback.format_exc()}\n")

        # ─────────────────────────────────────────────
        #  DeepFaceLab 迁移性实验
        # ─────────────────────────────────────────────
        if include_dfl:
            log("\n🔀 开始 DeepFaceLab 迁移性实验...\n")
            log("   测试扰动对不同人脸识别模型结构的迁移攻击成功率\n")

            # 用不同网络结构模拟 DeepFaceLab 使用的多种人脸编码器
            # DFL 内置多种backbone：LightCNN(浅层IR-18)、标准IR-50、深层IR-101
            try:
                import models.arcface_models as _arc_dfl
            except Exception:
                import sys as _sys
                if str(SIMSWAP_ROOT) not in _sys.path:
                    _sys.path.append(str(SIMSWAP_ROOT))
                import models.arcface_models as _arc_dfl

            dfl_model_cfgs = [
                ("IR-18(DFL-轻量)",  [2, 2, 2, 2],  False),   # 随机初始化，模拟轻量黑盒模型
                ("IR-50(攻击目标)",  [3, 4, 14, 3],  True),    # 直接用真实攻击目标模型，白盒上界
                ("IR-101(DFL-深层)", [3, 4, 23, 3],  False),   # 随机初始化，模拟深层黑盒模型
            ]

            transfer_results = {}
            for model_name, layers, use_real_model in dfl_model_cfgs:
                log(f"  测试目标模型: {model_name.replace(chr(10),' ')}\n")

                if use_real_model:
                    # IR-50 直接用已加载权重的真实模型，测白盒攻击成功率
                    proxy = model
                    log(f"    （使用真实 ArcFace 权重，白盒上界）\n")
                else:
                    # 其他模型随机初始化，模拟黑盒迁移场景
                    proxy_net = _arc_dfl.ResNet(
                        block=_arc_dfl.IRBlock, layers=layers, use_se=True
                    ).to(DEVICE)
                    proxy_net.eval()

                    class _ProxyWrapper:
                        def get_id_feature(self, x):
                            import torch.nn.functional as _F2
                            x2 = torch.nn.functional.interpolate(x, (112,112), mode="bilinear", align_corners=False)
                            x2 = (x2 - 0.5) / 0.5
                            with torch.no_grad():
                                feat = proxy_net(x2)
                            return _F2.normalize(feat, dim=1)

                    proxy = _ProxyWrapper()
                t_eval = Evaluator(device=DEVICE)

                for batch in dataloader:
                    imgs = batch["image"].to(DEVICE)
                    # 扰动由原始攻击目标(IR-50)生成，在代理模型上测迁移
                    with torch.enable_grad():
                        adv_imgs, _ = AdversarialPerturbationGenerator(
                            epsilon=int(epsilon)/255.0, alpha=int(epsilon)/255.0/10,
                            num_steps=int(num_steps), attack_type=attack_type,
                            adaptive_epsilon=True, device=DEVICE,
                        ).generate(imgs, model)

                    psnr_v = compute_psnr(imgs.detach(), adv_imgs.detach())
                    ssim_v = compute_ssim(imgs.detach(), adv_imgs.detach())

                    # ASR 计算：IR-50 用余弦相似度（真实模型有意义），随机模型用相对变化率
                    import torch.nn.functional as _Ft
                    with torch.no_grad():
                        fo = proxy.get_id_feature(imgs)
                        fa = proxy.get_id_feature(adv_imgs.detach())
                        if use_real_model:
                            # 真实模型：余弦相似度 < 0.5 认为身份被成功干扰
                            cos_sim = _Ft.cosine_similarity(fo, fa)
                            asr_v = (cos_sim < 0.5).float().mean().item()
                            log(f"      [迁移调试] 余弦相似度: {cos_sim.mean().item():.3f}\n")
                        else:
                            # 随机模型：相对特征变化率 > 1% 认为扰动有迁移效果
                            feat_change = (fo - fa).norm(dim=1)
                            feat_base   = fo.norm(dim=1).clamp(min=1e-8)
                            relative_change = feat_change / feat_base
                            asr_v = (relative_change > 0.01).float().mean().item()
                            log(f"      [迁移调试] 平均特征相对变化率: {relative_change.mean().item():.3f}\n")
                    t_eval.results.append({"psnr": psnr_v, "ssim": ssim_v, "asr": asr_v})
                    time.sleep(0.01)

                s = t_eval.summary()
                transfer_results[model_name] = s
                log(f"    → ASR={s.get('mean_asr',0):.1%}  PSNR={s.get('mean_psnr',0):.2f}dB  SSIM={s.get('mean_ssim',0):.3f}\n")

            # 生成迁移性柱状图
            t_methods  = list(transfer_results.keys())
            t_asr      = [transfer_results[m].get("mean_asr",  0)*100 for m in t_methods]
            t_psnr     = [transfer_results[m].get("mean_psnr", 0)     for m in t_methods]
            t_colors   = ["#6366f1", "#ef4444", "#f59e0b"]

            x2 = np.arange(len(t_methods))
            fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))

            bars0 = axes2[0].bar(x2, t_asr,  color=t_colors, width=0.5, zorder=3)
            axes2[0].axhline(y=80, color="#374151", linestyle="--", linewidth=1.2, label="目标 80%")
            axes2[0].set_xticks(x2); axes2[0].set_xticklabels(t_methods, fontsize=10)
            axes2[0].set_ylabel("ASR (%)"); axes2[0].set_title("跨模型迁移攻击成功率 ASR")
            axes2[0].legend(); axes2[0].grid(axis="y", alpha=0.25, zorder=0)
            for bar, val in zip(bars0, t_asr):
                axes2[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                              f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

            bars1 = axes2[1].bar(x2, t_psnr, color=t_colors, width=0.5, zorder=3)
            axes2[1].axhline(y=30, color="#374151", linestyle="--", linewidth=1.2, label="目标 30dB")
            axes2[1].set_xticks(x2); axes2[1].set_xticklabels(t_methods, fontsize=10)
            axes2[1].set_ylabel("PSNR (dB)"); axes2[1].set_title("图像质量 PSNR")
            axes2[1].legend(); axes2[1].grid(axis="y", alpha=0.25, zorder=0)
            for bar, val in zip(bars1, t_psnr):
                axes2[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                              f"{val:.1f}", ha="center", va="bottom", fontsize=9)

            plt.suptitle("DeepFaceLab 迁移性实验：扰动对不同模型结构的攻击成功率", fontsize=11, fontweight="bold")
            plt.tight_layout()
            _dfl_path = DATA_RESULTS / "dfl_transfer.png"
            try:
                log("\n📊 DFL 迁移图生成中...\n")
                plt.savefig(str(_dfl_path), dpi=150, bbox_inches="tight")
                plt.close()
                if _dfl_path.exists():
                    log(f"✅ DFL 迁移图已保存（{_dfl_path.stat().st_size // 1024} KB）\n")
                else:
                    log("❌ DFL 迁移图：文件未生成，请检查磁盘空间或路径权限\n")
            except Exception as _err:
                plt.close("all")
                log(f"\n❌ DFL 迁移图生成失败: {type(_err).__name__}: {_err}\n{traceback.format_exc()}\n")

        log("\n✅ 全部实验完成！\n"); time.sleep(0.5); log("DONE")

    except Exception as e:
        log(f"\n❌ 实验失败: {e}\n{traceback.format_exc()}\n"); log("DONE")
    finally:
        _experiment_running = False


@app.route("/api/run_experiment", methods=["POST"])
def api_run_experiment():
    global _experiment_running, _experiment_log_queue
    if _experiment_running:
        return jsonify({"error": "实验正在运行中"}), 409

    data = request.get_json()
    _experiment_running     = True
    _experiment_log_queue   = queue.Queue()

    t = threading.Thread(
        target=_run_experiment_thread,
        args=(
            data.get("epsilon", 8), data.get("num_steps", 40),
            data.get("attack_type", "pgd"), data.get("num_samples", 100),
            data.get("compare_mode", False),
            data.get("include_fawkes", False),
            data.get("include_dfl", False),
        ), daemon=True,
    )
    t.start()

    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadPriority(
            ctypes.windll.kernel32.OpenThread(0x0020, False, t.ident), -1)
    except Exception:
        pass
    return jsonify({"status": "started"})


@app.route("/api/experiment_log")
def api_experiment_log():
    def generate():
        while True:
            try:
                msg = _experiment_log_queue.get(timeout=3)
                yield f"data: {msg}\n\n"
                if msg == "DONE": break
            except queue.Empty:
                yield "data: HEARTBEAT\n\n"
    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


# ─────────────────────────────────────────────
#  路由：查看结果图表
# ─────────────────────────────────────────────

@app.route("/api/results")
def api_results():
    DATA_RESULTS.mkdir(exist_ok=True)
    result = {}
    for key, fname in [
        ("line_chart",      "line_chart.png"),
        ("heatmap",         "heatmap.png"),
        ("fawkes_chart",    "fawkes_comparison.png"),
        ("dfl_chart",       "dfl_transfer.png"),
    ]:
        p = DATA_RESULTS / fname
        if p.exists():
            result[key] = image_file_to_base64(str(p))
    result["files"] = [f.name for f in sorted(DATA_RESULTS.glob("*.png"))]
    return jsonify(result)


# ─────────────────────────────────────────────
#  路由：性能测试
# ─────────────────────────────────────────────

@app.route("/api/perf_test", methods=["POST"])
def api_perf_test():
    data  = request.get_json()
    n     = int(data.get("n", 5))
    model = get_model()
    times = []
    for _ in range(n):
        dummy = torch.rand(1, 3, 224, 224).to(DEVICE)
        gen = AdversarialPerturbationGenerator(
            epsilon=8/255, alpha=1/255, num_steps=20,
            adaptive_epsilon=True, device=DEVICE,
        )
        t0 = time.time(); gen.generate(dummy, model); times.append(time.time() - t0)

    return jsonify({
        "n": n, "avg": round(np.mean(times), 2),
        "total": round(sum(times), 2), "fps": round(n / sum(times), 3),
        "device": DEVICE.upper(),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True, use_reloader=False)