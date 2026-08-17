#!/usr/bin/env python3
"""Stage 0 check: PyTorch + CUDA on the target GPU (RTX 5060 Ti / Blackwell sm_120).

Exit codes: 0 = PASS, 1 = FAIL.
Run standalone:  python scripts/checks/check_cuda.py
"""
import sys


def fail(msg: str) -> int:
    print(f"[FAIL] cuda: {msg}")
    return 1


def main() -> int:
    try:
        import torch
    except ImportError:
        return fail(
            "torch is not installed. Run:\n"
            "       pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128"
        )

    print(f"       torch={torch.__version__}  cuda_build={torch.version.cuda}")

    if not torch.cuda.is_available():
        return fail(
            "torch.cuda.is_available() == False.\n"
            "       Common causes:\n"
            "       (1) a CPU-only wheel -> reinstall with --index-url .../cu128 and --no-cache-dir;\n"
            "       (2) the NVIDIA driver on Windows is missing or too old "
            "(WSL2 uses the Windows driver; try 'nvidia-smi');\n"
            "       (3) you are running outside WSL2."
        )

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    try:
        x = torch.randn(512, 512, device="cuda")
        _ = (x @ x).sum().item()
        torch.cuda.synchronize()
    except RuntimeError as e:
        return fail(
            f"the GPU is visible ({name}, sm_{cap[0]}{cap[1]}) but running a kernel "
            f"failed: {e}\n"
            "       -> this torch wheel does not support sm_120 (Blackwell). Install the "
            "latest stable build\n"
            "          from the cu128 index or newer, and do NOT reuse an old wheel "
            "from the pip cache."
        )

    warn = "" if "5060" in name else "  (not a 5060 Ti - still acceptable)"
    print(f"[PASS] cuda: {name} (sm_{cap[0]}{cap[1]}), GPU matmul OK{warn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
