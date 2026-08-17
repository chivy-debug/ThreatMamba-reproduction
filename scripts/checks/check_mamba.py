#!/usr/bin/env python3
"""Stage 0 check for the SSM layer: prefer the real mamba-ssm kernel, accept the fallback.

Exit codes: 0 = PASS (including a fallback pass), 1 = FAIL (both paths broken).
Run standalone:  python scripts/checks/check_mamba.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def fail(msg: str) -> int:
    print(f"[FAIL] mamba: {msg}")
    return 1


def try_real_mamba():
    """Returns (ok, info). Never raises."""
    try:
        import torch
        from mamba_ssm import Mamba
    except ImportError as e:
        return False, f"mamba-ssm is not installed ({type(e).__name__})"
    if not torch.cuda.is_available():
        return False, "mamba-ssm is present but torch cannot see a GPU"
    try:
        m = Mamba(d_model=48, d_state=16, d_conv=4, expand=2).to("cuda")
        x = torch.randn(2, 64, 48, device="cuda")
        t0 = time.time()
        y = m(x)
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000
    except Exception as e:  # noqa: BLE001
        hint = ""
        if "no kernel image" in str(e).lower():
            cap = torch.cuda.get_device_capability(0)
            hint = (f" -> the installed mamba-ssm has NO kernel for sm_{cap[0]}{cap[1]}. "
                    "Prebuilt wheels only go up to sm_100; build from source: "
                    "bash scripts/setup_mamba.sh")
        return False, f"{type(e).__name__}: {e}{hint}"
    if tuple(y.shape) != (2, 64, 48):
        return False, f"wrong shape: {tuple(y.shape)}"
    return True, f"real kernel OK, forward (2,64,48) in {dt:.1f} ms"


def try_fallback():
    try:
        import torch

        from src.ssm import SSMStack
    except ImportError as e:
        return False, f"failed to import the fallback: {e}"
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        s = SSMStack(48, n_layers=2, mode="simple").to(dev)
        y = s(torch.randn(2, 32, 48, device=dev), torch.ones(2, 32, device=dev))
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    return (tuple(y.shape) == (2, 32, 48)), f"SimpleSSM OK on {dev}"


def main() -> int:
    forced = os.getenv("THREATMAMBA_SSM")
    if forced == "simple":
        ok, info = try_fallback()
        if not ok:
            return fail(f"THREATMAMBA_SSM=simple was forced but the fallback is "
                        f"broken: {info}")
        print(f"[PASS] mamba: FALLBACK mode as requested (THREATMAMBA_SSM=simple) - {info}")
        print("       Note this in any write-up: the original mamba-ssm kernel is not used.")
        return 0

    ok, info = try_real_mamba()
    if ok:
        print(f"[PASS] mamba: {info}")
        return 0

    ok2, info2 = try_fallback()
    if not ok2:
        return fail(f"both paths are broken.\n       mamba-ssm: {info}\n"
                    f"       fallback : {info2}")

    print(f"[PASS] mamba: FALLBACK - {info2}")
    print(f"       (mamba-ssm is unusable: {info})")
    print("       Stages 1-7 all run in this mode; just set:  export THREATMAMBA_SSM=simple")
    print("       To use the paper's real kernel:  bash scripts/setup_mamba.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
