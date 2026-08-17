#!/usr/bin/env python3
"""Patch the CUDA headers for glibc >= 2.41 (Ubuntu 25.10/26.04, Fedora 42+, ...).

THE PROBLEM
  glibc 2.41 added cospi/sinpi/cospif/sinpif/rsqrt/rsqrtf to <math.h> declared
  `noexcept (true)`. The CUDA header `crt/math_functions.h` declares the same names
  WITHOUT an exception specification, so nvcc reports:
      error: exception specification is incompatible with that of previous function
  This affects EVERY CUDA 12.x release and 13.x as well (rsqrt is still unpatched
  upstream).

THE PATCH
  Append ` noexcept (true)` to exactly those six declarations in the CUDA header. This is
  safe: it only brings them into agreement with glibc.

USAGE
  python3 scripts/patch_cuda_glibc.py --check          # check only, change nothing
  sudo python3 scripts/patch_cuda_glibc.py             # patch (writes a .bak_tm backup)
  sudo python3 scripts/patch_cuda_glibc.py --restore   # revert

Exit codes: 0 = patched or no patch needed, 1 = failed.
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

FUNCS = ["cospi", "cospif", "sinpi", "sinpif", "rsqrt", "rsqrtf"]
BACKUP_SUFFIX = ".bak_tm"


def find_headers() -> list[Path]:
    roots = []
    if os.getenv("CUDA_HOME"):
        roots.append(Path(os.environ["CUDA_HOME"]))
    roots += [Path("/usr/local/cuda"), *sorted(Path("/usr/local").glob("cuda-*"))]
    out, seen = [], set()
    for r in roots:
        for rel in ("include/crt/math_functions.h",
                    "targets/x86_64-linux/include/crt/math_functions.h"):
            p = (r / rel)
            if p.exists():
                real = p.resolve()
                if real not in seen:
                    seen.add(real)
                    out.append(real)
    return out


def patch_text(s: str) -> tuple[str, int]:
    n = 0
    for fn in FUNCS:
        # declarations look like:  extern ... double rsqrt(double x);   (no noexcept/throw yet)
        pat = re.compile(
            r"^([ \t]*extern[^\n;]*\b(?:double|float)[ \t]+" + fn + r"[ \t]*\([^;\n]*\))[ \t]*;",
            re.M)

        def repl(m):
            nonlocal n
            line = m.group(1)
            if "noexcept" in line or "__THROW" in line or "throw" in line:
                return m.group(0)
            n += 1
            return line + " noexcept (true);"

        s = pat.sub(repl, s)
    return s, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    headers = find_headers()
    if not headers:
        print("[patch] crt/math_functions.h not found - is CUDA installed?")
        return 1

    for h in headers:
        bak = h.with_suffix(h.suffix + BACKUP_SUFFIX)
        if args.restore:
            if bak.exists():
                shutil.copy2(bak, h)
                print(f"[patch] reverted: {h}")
            else:
                print(f"[patch] no backup found for {h}")
            continue

        text = h.read_text(encoding="utf-8", errors="surrogateescape")
        new, n = patch_text(text)
        if n == 0:
            print(f"[patch] {h}: no patch needed (already noexcept, or declarations not found)")
            continue
        if args.check:
            print(f"[patch] {h}: {n} declarations NEED patching (re-run with sudo, without --check)")
            continue
        if not bak.exists():
            shutil.copy2(h, bak)
        try:
            h.write_text(new, encoding="utf-8", errors="surrogateescape")
        except PermissionError:
            print(f"[patch] no write permission for {h} - re-run with sudo")
            return 1
        print(f"[patch] {h}: patched {n} declarations (backup: {bak.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
