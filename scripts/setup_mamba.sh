#!/usr/bin/env bash
# ==========================================================================
# Build causal-conv1d + mamba-ssm FROM SOURCE for RTX 5060 Ti (Blackwell, sm_120).
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MAMBA_TAG="v2.2.6"
CAUSAL_TAG="v1.6.2.post1"
CUDA_APT_VER="12-8"
ALL_ARCH=0; CHECK_ONLY=0
for a in "$@"; do
  case "$a" in
    --all-arch) ALL_ARCH=1 ;;
    --check)    CHECK_ONLY=1 ;;
    *) echo "unknown argument: $a"; exit 1 ;;
  esac
done
export MAX_JOBS="${MAX_JOBS:-4}"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[ -x .venv/bin/python ] || die "no .venv found - run 'bash scripts/setup_env.sh' first"
# shellcheck disable=SC1091
source .venv/bin/activate
python -c "import torch" 2>/dev/null || die "torch is not installed in .venv"

TORCH_VER=$(python -c "import torch;print(torch.__version__.split('+')[0])")
TORCH_CUDA=$(python -c "import torch;print(torch.version.cuda)")
ABI=$(python -c "import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)")
GLIBC=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo "?")
say "Current environment"
echo "    torch $TORCH_VER (cuda $TORCH_CUDA, cxx11abi=$ABI) | glibc $GLIBC | MAX_JOBS=$MAX_JOBS"

# ---------------------------------------------------------------- [1] nvcc >= 12.8
say "[1/6] CUDA Toolkit (nvcc >= 12.8)"
nvcc_ver() { "$1" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1; }
NVCC=""
for c in "${CUDA_HOME:-/nonexistent}/bin/nvcc" /usr/local/cuda/bin/nvcc \
         /usr/local/cuda-12.9/bin/nvcc /usr/local/cuda-12.8/bin/nvcc "$(command -v nvcc || true)"; do
  [ -x "$c" ] || continue
  v=$(nvcc_ver "$c"); [ -n "$v" ] || continue
  if [ "$(printf '%s\n12.8\n' "$v" | sort -V | head -1)" = "12.8" ]; then NVCC="$c"; break; fi
  echo "    bo qua $c (CUDA $v < 12.8)"
done
if [ -z "$NVCC" ]; then
  echo "    Installing a minimal CUDA Toolkit from the NVIDIA repository..."
  if ! grep -rq "developer.download.nvidia.com" /etc/apt/sources.list.d/ 2>/dev/null; then
    TMPD=$(mktemp -d)
    curl -fsSL -o "$TMPD/keyring.deb" \
      https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i "$TMPD/keyring.deb"; rm -rf "$TMPD"
  fi
  echo 'Acquire::ForceIPv4 "true";' | sudo tee /etc/apt/apt.conf.d/99force-ipv4 > /dev/null
  sudo apt-get update -y
  sudo apt-get install -y "cuda-nvcc-$CUDA_APT_VER" "cuda-cudart-dev-$CUDA_APT_VER" \
                          "cuda-cccl-$CUDA_APT_VER"
  NVCC="/usr/local/cuda-${CUDA_APT_VER/-/.}/bin/nvcc"
  [ -x "$NVCC" ] || NVCC="/usr/local/cuda/bin/nvcc"
  [ -x "$NVCC" ] || die "installation finished but nvcc was not found"
fi
export CUDA_HOME; CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "    using nvcc $(nvcc_ver "$NVCC") at $CUDA_HOME"

# ---------------------------------------------------------------- [2] host compiler <= 14
say "[2/6] Host compiler (CUDA 12.8 requires g++ <= 14)"
GXX_MAJOR=$(g++ -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)
HOSTCXX=""
if [ "${GXX_MAJOR:-0}" -ge 1 ] && [ "${GXX_MAJOR:-0}" -le 14 ]; then
  HOSTCXX="$(command -v g++)"
else
  for v in 14 13 12; do
    command -v "g++-$v" >/dev/null 2>&1 && { HOSTCXX="$(command -v g++-$v)"; break; }
  done
  if [ -z "$HOSTCXX" ]; then
    echo "    system g++ is $GXX_MAJOR (>14) - installing g++-14/g++-13…"
    sudo apt-get install -y g++-14 gcc-14 2>/dev/null || sudo apt-get install -y g++-13 gcc-13 2>/dev/null || true
    for v in 14 13; do
      command -v "g++-$v" >/dev/null 2>&1 && { HOSTCXX="$(command -v g++-$v)"; break; }
    done
  fi
  [ -n "$HOSTCXX" ] || die "no g++ <= 14 available. Use the fallback: export THREATMAMBA_SSM=simple"
fi
export CXX="$HOSTCXX"
export CC="${HOSTCXX/g++/gcc}"
export CUDAHOSTCXX="$HOSTCXX"
unset NVCC_PREPEND_FLAGS 2>/dev/null || true
echo "    host compiler: $HOSTCXX ($($HOSTCXX -dumpversion))"

# ---------------------------------------------------------------- [3] TEST BIEN DICH THU
say "[3/6] Trial compilation"
TESTDIR=$(mktemp -d); trap 'rm -rf "$TESTDIR"' EXIT
cat > "$TESTDIR/t.cu" <<'CUEOF'
#include <math.h>
#include <cmath>
__global__ void k(float* o) { o[0] = 1.0f; }
int main() { return 0; }
CUEOF
try_compile() {
  "$NVCC" -std=c++17 -ccbin "$HOSTCXX" -gencode arch=compute_120,code=sm_120 \
          -c "$TESTDIR/t.cu" -o "$TESTDIR/t.o" 2> "$TESTDIR/err.log"
}
if try_compile; then
  echo "    OK - nvcc compiles successfully"
else
  if grep -q "exception specification is incompatible" "$TESTDIR/err.log"; then
    echo "    Hit the glibc $GLIBC / CUDA header conflict (cospi/sinpi/rsqrt)."
    sudo -E env "CUDA_HOME=$CUDA_HOME" python3 scripts/patch_cuda_glibc.py \
      || die "patching the headers failed"
    if try_compile; then
      echo "    OK - nvcc compiles after the patch"
    else
      echo "--- first 20 error lines ---"; head -20 "$TESTDIR/err.log"
      die "still cannot compile after patching. Use the fallback: export THREATMAMBA_SSM=simple"
    fi
  else
    echo "--- first 20 error lines ---"; head -20 "$TESTDIR/err.log"
    die "nvcc cannot compile (not the glibc issue). See the log above."
  fi
fi

if [ "$CHECK_ONLY" = 1 ]; then
  echo; echo "=== --check: the environment is READY to build. ==="
  exit 0
fi

# ---------------------------------------------------------------- [4] goi phu tro
say "[4/6] Support packages"
pip install --no-cache-dir einops ninja packaging

BUILD_DIR="${TMPDIR:-/tmp}/tm_build"
mkdir -p "$BUILD_DIR"

# Replace the removed gencode block with 'pass' to avoid an IndentationError.
patch_arch() {
  local f="$1"
  [ "$ALL_ARCH" = 1 ] && { echo "    (keeping every architecture)"; return; }
  python - "$f" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
pat = re.compile(
    r'(?P<ind>[ \t]*)cc_flag\.append\((?P<q>["\'])-gencode(?P=q)\)\n'
    r'[ \t]*cc_flag\.append\((?P<q2>["\'])arch=compute_(?P<n>\d+),code=sm_\d+(?P=q2)\)'
)
kept = []
def repl(m):
    if m.group("n") == "120":
        kept.append("120")
        return m.group(0)
    # Substitute a 'pass' statement with the original indentation to keep valid Python
    return f"{m.group('ind')}pass"

s2, n = pat.subn(repl, s)
open(p, "w").write(s2)
left = re.findall(r"arch=compute_(\d+),code=sm_", s2)
print(f"    patch {p.split('/')[-2]}/setup.py: kept gencode {sorted(set(left))}")
if "120" not in left:
    print("    ! WARNING: no sm_120 gencode found")
PYEOF
}

build_one() {
  local name="$1" repo="$2" tag="$3" forcevar="$4"
  say "Build $name ($tag)"
  rm -rf "$BUILD_DIR/$name"
  git clone -q --depth 1 --branch "$tag" "$repo" "$BUILD_DIR/$name"
  patch_arch "$BUILD_DIR/$name/setup.py"
  ( cd "$BUILD_DIR/$name"
    export "$forcevar"=TRUE
    # Force torch to compile for sm_120 only
    export TORCH_CUDA_ARCH_LIST="12.0"
    pip install --no-build-isolation --no-cache-dir . )
}

say "[5/6] causal-conv1d + mamba-ssm (build takes 10-40 minutes)"
build_one causal-conv1d https://github.com/Dao-AILab/causal-conv1d "$CAUSAL_TAG" CAUSAL_CONV1D_FORCE_BUILD
build_one mamba          https://github.com/state-spaces/mamba      "$MAMBA_TAG"  MAMBA_FORCE_BUILD

say "[6/6] Verify the real kernel on the GPU"
python - <<'PYEOF'
import sys, torch
try:
    from mamba_ssm import Mamba
except ImportError as e:
    print(f"    FAIL: importing mamba_ssm failed: {e}"); sys.exit(1)
if not torch.cuda.is_available():
    print("    FAIL: no GPU visible"); sys.exit(1)
cap = torch.cuda.get_device_capability(0)
m = Mamba(d_model=48, d_state=16, d_conv=4, expand=2).cuda()
y = m(torch.randn(2, 64, 48, device="cuda")); torch.cuda.synchronize()
assert tuple(y.shape) == (2, 64, 48), y.shape
print(f"    PASS: mamba-ssm runs natively on {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
PYEOF

cat <<'EOF'

==========================================================
Done. THREATMAMBA_SSM no longer needs to be set.
==========================================================
EOF