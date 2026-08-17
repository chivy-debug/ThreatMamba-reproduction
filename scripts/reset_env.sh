#!/usr/bin/env bash
# ==========================================================================
# Remove everything this project installed, so you can start over.
#
#   bash scripts/reset_env.sh            # project-scope cleanup (venv, caches) - safe
#   bash scripts/reset_env.sh --deep     # + remove the apt CUDA toolkit, pip/HF caches
#   bash scripts/reset_env.sh --deep --yes
#
# NEVER touches: downloaded data (data/), results (outputs/), or Ollama on Windows.
# ==========================================================================
set -u
cd "$(dirname "$0")/.."
DEEP=0; YES=0
for a in "$@"; do
  case "$a" in
    --deep) DEEP=1 ;;
    --yes|-y) YES=1 ;;
    *) echo "unknown argument: $a"; exit 1 ;;
  esac
done

confirm() {
  [ "$YES" = 1 ] && return 0
  read -r -p "$1 [y/N] " r; [ "$r" = y ] || [ "$r" = Y ]
}

echo "== [1] Project Python environment =="
for d in .venv venv env; do
  [ -d "$d" ] && { rm -rf "$d"; echo "   removed ./$d"; }
done
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
rm -rf .pytest_cache build dist *.egg-info 2>/dev/null
echo "   removed __pycache__ / build artifacts"

echo "== [2] Temporary build directories for mamba / causal-conv1d =="
rm -rf ~/.cache/torch_extensions /tmp/tm_build /tmp/mamba /tmp/causal-conv1d 2>/dev/null
echo "   done"

if [ "$DEEP" = 1 ]; then
  echo "== [3] pip + Hugging Face caches (can be several GB) =="
  if confirm "   Delete ~/.cache/pip and ~/.cache/huggingface?"; then
    rm -rf ~/.cache/pip ~/.cache/huggingface ~/.cache/uv
    echo "   deleted"
  fi

  echo "== [4] CUDA toolkit installed via apt (12.4 causes conflicts) =="
  if dpkg -l 2>/dev/null | grep -qE "^ii\s+(nvidia-cuda-toolkit|cuda-toolkit)"; then
    if confirm "   Purge nvidia-cuda-toolkit / cuda-* via apt?"; then
      sudo apt-get purge -y 'nvidia-cuda-toolkit*' 'cuda-toolkit*' 'cuda-[0-9]*' 2>/dev/null
      sudo apt-get autoremove -y
      echo "   purged"
    fi
  else
    echo "   no apt-installed CUDA packages found - skipping"
  fi

  echo "== [5] Ollama installed inside WSL (remove it if you use Ollama on Windows) =="
  if command -v ollama >/dev/null 2>&1; then
    if confirm "   Remove Ollama from WSL (including models under ~/.ollama)?"; then
      sudo systemctl stop ollama 2>/dev/null
      sudo systemctl disable ollama 2>/dev/null
      sudo rm -f /etc/systemd/system/ollama.service /usr/local/bin/ollama /usr/bin/ollama
      rm -rf ~/.ollama
      echo "   removed"
    fi
  else
    echo "   Ollama is not installed in WSL - skipping"
  fi
fi

echo
echo "=========================================================="
echo "Cleanup complete. Reinstall with:   bash scripts/setup_env.sh"
echo
echo "For a TOTAL reset (removing the whole Ubuntu distro), run this from"
echo "PowerShell on Windows - NOTE: this destroys everything inside WSL:"
echo "   wsl --list --verbose"
echo "   wsl --unregister Ubuntu-24.04     # use the exact distro name listed above"
echo "   wsl --install -d Ubuntu-24.04"
echo "=========================================================="
