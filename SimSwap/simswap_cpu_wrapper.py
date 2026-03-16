"""
SimSwap CPU 推理包装脚本
放置位置：SimSwap 仓库根目录（与 test_one_image.py 同级）
"""

import os
import sys

# 步骤 1：屏蔽 CUDA 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn

# 步骤 2：patch torch.device
_orig_device = torch.device

class _ForceCPUDevice:
    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], str) and "cuda" in args[0]:
            return _orig_device("cpu")
        return _orig_device(*args, **kwargs)

torch.device = _ForceCPUDevice

# 步骤 3：patch Tensor.to()
# fs_networks.py 里有 tensor.to(device)，device 是 cuda，
# 导致模型权重(cpu) vs 中间张量(cuda) 不一致 -> addmm 报错
_orig_tensor_to = torch.Tensor.to

def _cpu_tensor_to(self, *args, **kwargs):
    new_args = []
    for a in args:
        if isinstance(a, str) and "cuda" in a:
            new_args.append("cpu")
        elif isinstance(a, torch.device) and a.type == "cuda":
            new_args.append(torch.device("cpu"))
        else:
            new_args.append(a)
    if "device" in kwargs:
        d = kwargs["device"]
        if isinstance(d, str) and "cuda" in d:
            kwargs["device"] = "cpu"
        elif isinstance(d, torch.device) and d.type == "cuda":
            kwargs["device"] = torch.device("cpu")
    return _orig_tensor_to(self, *new_args, **kwargs)

torch.Tensor.to = _cpu_tensor_to

# 步骤 4：patch Module.to()
_orig_module_to = nn.Module.to

def _cpu_module_to(self, *args, **kwargs):
    new_args = []
    for a in args:
        if isinstance(a, str) and "cuda" in a:
            new_args.append("cpu")
        elif isinstance(a, torch.device) and a.type == "cuda":
            new_args.append(torch.device("cpu"))
        else:
            new_args.append(a)
    if "device" in kwargs:
        d = kwargs["device"]
        if isinstance(d, str) and "cuda" in d:
            kwargs["device"] = "cpu"
        elif isinstance(d, torch.device) and d.type == "cuda":
            kwargs["device"] = torch.device("cpu")
    return _orig_module_to(self, *new_args, **kwargs)

nn.Module.to = _cpu_module_to

# 步骤 5：patch .cuda()
torch.Tensor.cuda = lambda self, *a, **kw: self.cpu()
nn.Module.cuda   = lambda self, *a, **kw: self.cpu()

# 步骤 6：patch torch.cuda.is_available
torch.cuda.is_available = lambda: False

# 步骤 7：patch torch.load（旧版不支持 map_location=torch.device(...)）
_orig_load = torch.load

def _cpu_load(f, map_location=None, pickle_module=None, **kwargs):
    map_location = "cpu"
    if pickle_module is not None:
        return _orig_load(f, map_location=map_location,
                          pickle_module=pickle_module, **kwargs)
    return _orig_load(f, map_location=map_location, **kwargs)

torch.load = _cpu_load

# 步骤 8：执行 test_one_image.py
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_test_script = os.path.join(_script_dir, "test_one_image.py")

with open(_test_script, "r", encoding="utf-8") as _f:
    _code = _f.read()

exec(compile(_code, _test_script, "exec"),
     {"__name__": "__main__", "__file__": _test_script})