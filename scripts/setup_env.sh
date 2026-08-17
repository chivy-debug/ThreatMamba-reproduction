#!/usr/bin/env bash
# ==========================================================================
# ThreatMAMBA reproduction - Stage 0 environment setup (run INSIDE WSL2 Ubuntu or Linux)
#
#   bash scripts/setup_env.sh
#
# Result: a .venv with Python 3.11 + PyTorch cu128 (works on RTX 5060 Ti / sm_120)
#         and every library needed for Stages 1-7.
#
# mamba-ssm is deliberately NOT installed here:
#   - the prebuilt mamba-ssm wheels are NOT compiled for sm_120, so they fail on a 5060 Ti
#   - building from source needs nvcc >= 12.8 and g++ <= 14; that is a separate step:
#         bash scripts/setup_mamba.sh
#   - meanwhile the project runs end to end with the SSM fallback:
#         export THREATMAMBA_SSM=simple
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY_TARGET="3.11"          # best-supported Python for this ecosystem (3.14 has no wheels yet)
TORCH_VER="2.8.0"         # torch version with matching mamba-ssm / causal-conv1d wheels
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Linux" ] || die "this must run inside WSL2 Ubuntu or Linux (uname: $(uname -s))"
case "$PWD" in
  /mnt/*) echo "WARNING: the repository is on a Windows drive ($PWD) - I/O is slow and builds often fail."
          echo "         Better: cp -r \"$PWD\" ~/threatmamba && cd ~/threatmamba" ;;
esac

say "[1/6] System packages"
sudo apt-get update -y
sudo apt-get install -y build-essential git curl unzip ca-certificates \
                        python3-venv python3-dev

say "[2/6] Locate Python $PY_TARGET"
PYBIN=""
if command -v "python$PY_TARGET" >/dev/null 2>&1; then
  PYBIN="$(command -v python$PY_TARGET)"
  echo "   already present: $PYBIN"
else
  echo "   python$PY_TARGET not found - trying apt…"
  if sudo apt-get install -y "python$PY_TARGET" "python$PY_TARGET-venv" "python$PY_TARGET-dev" 2>/dev/null \
     && command -v "python$PY_TARGET" >/dev/null 2>&1; then
    PYBIN="$(command -v python$PY_TARGET)"
    echo "   installed via apt: $PYBIN"
  else
    echo "   apt has no python$PY_TARGET (Ubuntu too new) -> using uv to fetch a standalone build"
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "could not install uv. Install it manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
    uv python install "$PY_TARGET"
    PYBIN="$(uv python find "$PY_TARGET")"
    echo "   uv installed: $PYBIN"
  fi
fi
"$PYBIN" -c "import sys; assert sys.version_info[:2]==(3,11), sys.version" \
  || die "the Python found is not 3.11"

say "[3/6] Create .venv (Python $PY_TARGET)"
if [ -d .venv ]; then
  CUR="$(.venv/bin/python -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo none)"
  if [ "$CUR" != "$PY_TARGET" ]; then
    echo "   .venv uses Python $CUR -> removing and recreating"
    rm -rf .venv
  fi
fi
[ -d .venv ] || "$PYBIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -V
python -m pip install --upgrade pip wheel setuptools packaging ninja

say "[4/6] PyTorch $TORCH_VER + cu128 (Blackwell sm_120 support)"
pip install --no-cache-dir "torch==$TORCH_VER" --index-url "$TORCH_INDEX"

say "[5/6] Libraries for Stages 1-7"
pip install -r requirements.txt

say "[6/6] Quick check"
python - <<'PYEOF'
import torch
print(f"   torch      : {torch.__version__}  (cuda build {torch.version.cuda})")
ok = torch.cuda.is_available()
print(f"   cuda ready : {ok}")
if ok:
    name = torch.cuda.get_device_name(0); cap = torch.cuda.get_device_capability(0)
    print(f"   GPU        : {name}  sm_{cap[0]}{cap[1]}")
    x = torch.randn(512, 512, device="cuda"); (x @ x).sum().item()
    print("   GPU matmul : OK")
else:
    print("   ! torch cannot see a GPU - see the Troubleshooting section of the README")
PYEOF

cat <<'TXT'

==========================================================================
Base environment installed. Remaining Stage 0 steps:

  1) Ollama:
       - If Ollama is already installed on WINDOWS there is no need to reinstall it in WSL;
         just set OLLAMA_HOST=0.0.0.0:11434 on the Windows side.
       - To install inside WSL:  curl -fsSL https://ollama.com/install.sh | sh
         then:  ollama pull qwen3:8b
  2) API keys:   cp .env.example .env   and fill in VT_API_KEY, OTX_API_KEY
  3) Acceptance: source .venv/bin/activate && bash scripts/smoke_test.sh

About mamba-ssm (the paper's CUDA kernel):
  - It is NOT installed yet. The project runs all of Stages 1-7 in fallback mode:
        export THREATMAMBA_SSM=simple
  - To use the real kernel:  bash scripts/setup_mamba.sh
    That script installs nvcc >= 12.8 and g++ <= 14 and builds specifically for sm_120.
==========================================================================
TXT
