import os
import platform
import warnings
import traceback

try:
    _TF_RWKV_PY_OPS = bool(int(os.environ["TF_RWKV_PY_OPS"]))
except KeyError:
    if platform.system() == "Linux":
        _TF_RWKV_PY_OPS = False
    else:
        _TF_RWKV_PY_OPS = True

_FALLBACK_WARNING_TEMPLATE = """{}

The {} C++/CUDA custom op could not be loaded.
"""


def warn_fallback(op_name):
    warning_msg = _FALLBACK_WARNING_TEMPLATE.format(traceback.format_exc(), op_name)
    warnings.warn(warning_msg, RuntimeWarning)
    disable_custom_kernel()


def enable_custom_kernel():
    """Prefer custom C++/CUDA kernel to pure python operations.

    Enable using custom C++/CUDA kernel instead of pure python operations.
    It has the same effect as setting environment variable `TF_RWKV_PY_OPS=0`.
    """
    global _TF_RWKV_PY_OPS
    _TF_RWKV_PY_OPS = False


def disable_custom_kernel():
    """Prefer pure python operations to custom C++/CUDA kernel.

    Disable using custom C++/CUDA kernel instead of pure python operations.
    It has the same effect as setting environment variable `TF_RWKV_PY_OPS=1`.
    """
    global _TF_RWKV_PY_OPS
    _TF_RWKV_PY_OPS = True


def is_custom_kernel_disabled():
    """Return whether custom C++/CUDA kernel is disabled."""
    return _TF_RWKV_PY_OPS
