"""
人脸隐私保护系统 - Flask Web 后端
运行方式: python ui/app.py
访问: http://localhost:5000
"""

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
from models.simswap_wrapper import SimSwapWrapper
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
    str(SIMSWAP_ROOT / "crop_224" / "ds.jpg"),
)
# Python 解释器路径（可改为 conda 环境里的 python）
SIMSWAP_PYTHON = os.environ.get("SIMSWAP_PYTHON", sys.executable)

_swap_model: Optional[SimSwapWrapper] = None
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
        "--gpu_ids",   "-1",   # 强制 CPU，避免与主进程 cuDNN 上下文冲突
    ]

    # 隔离子进程的 CUDA 环境：令其看不到任何 GPU，彻底避免 cuDNN 初始化争抢
    sub_env = os.environ.copy()
    sub_env["CUDA_VISIBLE_DEVICES"] = ""

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(SIMSWAP_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=sub_env,
        )
        if proc.returncode != 0:
            print(f"[SimSwap] 子进程报错 (code={proc.returncode}):\n{proc.stderr[-1000:]}")
            return None

        result_path = os.path.join(output_dir, "result.jpg")
        if os.path.exists(result_path):
            return result_path
        else:
            print(f"[SimSwap] 未找到输出文件: {result_path}")
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

        # ── 换脸对比（官方 SimSwap）─────────────────────────
        # 取 numpy 格式的原图和保护图
        orig_np = tensor_to_numpy(img_tensor)
        adv_np  = tensor_to_numpy(adv_img.detach())

        swap_orig_b64 = swap_prot_b64 = None
        swap_mode = "simswap"  # 告诉前端用了真实换脸

        b_path = target_path_tmp if target_path_tmp else SIMSWAP_DEFAULT_TARGET

        if SIMSWAP_ROOT.exists() and (SIMSWAP_ROOT / "test_one_image.py").exists():
            # 调用官方 SimSwap ×2（原图 & 保护图各一次）
            with _tasks_lock:
                _protect_tasks[task_id]["progress"] = 60
                _protect_tasks[task_id]["status_hint"] = "SimSwap 换脸中（原图）..."

            swap_orig_b64, err1 = simswap_pair_to_base64(orig_np, b_path)

            with _tasks_lock:
                _protect_tasks[task_id]["progress"] = 80
                _protect_tasks[task_id]["status_hint"] = "SimSwap 换脸中（保护图）..."

            swap_prot_b64, err2 = simswap_pair_to_base64(adv_np, b_path)

            if swap_orig_b64 is None or swap_prot_b64 is None:
                # SimSwap 失败：退化为热力图可视化
                swap_mode = "heatmap"
                swap_orig_b64, swap_prot_b64 = _make_heatmap_pair(
                    img_tensor, adv_img, model, orig_np, adv_np
                )
        else:
            # SimSwap 未配置：使用热力图可视化
            swap_mode = "heatmap"
            swap_orig_b64, swap_prot_b64 = _make_heatmap_pair(
                img_tensor, adv_img, model, orig_np, adv_np
            )

        with _tasks_lock:
            _protect_tasks[task_id]["progress"] = 90

        result = {
            "original":       tensor_to_base64(img_tensor),
            "protected":      tensor_to_base64(adv_img),
            "perturbation":   tensor_to_base64((perturbation * 10 + 0.5).clamp(0, 1)),
            "epsilon_map":    tensor_to_base64(eps_vis),
            "swap_orig":      swap_orig_b64,
            "swap_protected": swap_prot_b64,
            "swap_mode":      swap_mode,       # "simswap" | "heatmap"
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
            _protect_tasks[task_id]["status"]   = "done"
            _protect_tasks[task_id]["progress"] = 100
            _protect_tasks[task_id]["result"]   = result

    except Exception as e:
        with _tasks_lock:
            _protect_tasks[task_id]["status"] = "error"
            _protect_tasks[task_id]["error"]  = str(e)
            _protect_tasks[task_id]["trace"]  = traceback.format_exc()

    finally:
        # 清理目标人脸临时文件
        if target_path_tmp and os.path.exists(target_path_tmp):
            os.unlink(target_path_tmp)


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


def get_model() -> SimSwapWrapper:
    global _swap_model
    if _swap_model is None:
        with _model_lock:
            if _swap_model is None:
                print(f"[Server] 加载 SimSwap 模型，设备: {DEVICE}")
                _swap_model = SimSwapWrapper(
                    model_path=str(SIMSWAP_ROOT / "checkpoints" / "people"),
                    img_size=224,
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

def _run_experiment_thread(epsilon, num_steps, attack_type, num_samples, compare_mode):
    global _experiment_running
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    def log(msg):
        _experiment_log_queue.put(msg)

    try:
        DATA_RESULTS.mkdir(parents=True, exist_ok=True)
        img_dir = DATA_PROCESSED / "images"
        if not img_dir.exists() or not list(img_dir.glob("*.png")):
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
                    elapsed      = time.time() - t0
                    target_imgs  = torch.roll(imgs, 1, 0)
                    metrics      = evaluator.evaluate_batch(model, imgs, adv_imgs, target_imgs, eps/255.0)
                    log(f"  Batch {b_idx}: PSNR={metrics['psnr']:.1f}dB  SSIM={metrics['ssim']:.3f}  ASR={metrics['asr']:.1%}  ({elapsed:.1f}s)\n")
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
            plt.savefig(str(DATA_RESULTS / "line_chart.png"), dpi=150, bbox_inches="tight")
            plt.close(); log("\n📊 折线图已生成\n")

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
            plt.savefig(str(DATA_RESULTS / "heatmap.png"), dpi=150, bbox_inches="tight")
            plt.close(); log("🗺️  热力图已生成\n")

        log("\n✅ 全部实验完成！\n"); log("DONE")

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
    for key, fname in [("line_chart", "line_chart.png"), ("heatmap", "heatmap.png")]:
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