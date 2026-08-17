#!/usr/bin/env bash
# ==========================================================================
# ThreatMAMBA reproduction - STAGE 0 ACCEPTANCE TEST
#
# Usage:        bash scripts/smoke_test.sh
# Debug mode:   SMOKE_OFFLINE=1 bash scripts/smoke_test.sh
#               (skips every check that needs network/API access - does NOT count as passing)
#
# PASSES when EVERY check passes (0 FAIL, 0 SKIP) -> exit 0.
# ==========================================================================
set -u
cd "$(dirname "$0")/.."

# ---- colours (disabled when not a tty) ----
if [ -t 1 ]; then G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[1m'; N=$'\e[0m'
else G=""; R=""; Y=""; B=""; N=""; fi

# ---- load .env if present ----
if [ -f .env ]; then set -a; . ./.env; set +a; echo "(.env loaded)"; fi
[ "${SMOKE_OFFLINE:-}" ] && echo "${Y}OFFLINE MODE: network checks will SKIP - this does not count as acceptance.${N}"

# ---- choose python: prefer .venv ----
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then PY=python3
else echo "${R}python3 not found.${N}"; exit 1; fi
echo "Using python: $("$PY" -c 'import sys; print(sys.executable)')"

PASS=(); FAIL=(); SKIP=()

run() {  # run <name> <command...>
  local name="$1"; shift
  printf '%s\n' "──────────────────────────────────────────────"
  printf '%s\n' "${B}[$name]${N}"
  "$@"
  local rc=$?
  case $rc in
    0) PASS+=("$name") ;;
    2) SKIP+=("$name") ;;
    *) FAIL+=("$name") ;;
  esac
}

# ---- 1. WSL2 / Linux environment ----
check_env() {
  case "$(uname -s)" in
    Linux) : ;;
    *) echo "[FAIL] env: not Linux/WSL2 (uname: $(uname -s)). Run this INSIDE WSL2 Ubuntu."; return 1 ;;
  esac
  local k; k=$(uname -r 2>/dev/null || echo "?")
  if printf '%s' "$k" | grep -qi microsoft; then
    echo "[PASS] env: WSL2 ($k)"
  else
    echo "[PASS] env: native Linux ($k) - WSL2 is the reference setup, but native Linux works too"
  fi
  return 0
}

# ---- 2. GPU driver (nvidia-smi inside WSL) ----
check_gpu_driver() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[FAIL] gpu-driver: nvidia-smi not found inside WSL."
    echo "       Install the LATEST NVIDIA driver on WINDOWS (WSL2 reuses it; do NOT install a"
    echo "       Linux driver inside WSL), then restart WSL: run 'wsl --shutdown' from PowerShell."
    return 1
  fi
  local info; info=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -n1)
  if [ -z "$info" ]; then echo "[FAIL] gpu-driver: nvidia-smi failed to run"; return 1; fi
  echo "[PASS] gpu-driver: $info"
  return 0
}

# ---- 3. Python 3.10-3.13 (3.14 has no wheels for the ML ecosystem yet) ----
check_pyver() {
  "$PY" - <<'PY_VER_EOF'
import sys
v = sys.version_info[:2]
if v == (3, 11):
    print(f"[PASS] python: {sys.version.split()[0]} (recommended version)")
    sys.exit(0)
if (3, 10) <= v <= (3, 13):
    print(f"[PASS] python: {sys.version.split()[0]} (workable; 3.11 recommended)")
    sys.exit(0)
print(f"[FAIL] python: {sys.version.split()[0]} - need 3.10-3.13 (3.11 is best).")
print("       Python 3.14 has no wheels for mamba-ssm/causal-conv1d or many ML libraries.")
print("       Fix: bash scripts/reset_env.sh && bash scripts/setup_env.sh")
sys.exit(1)
PY_VER_EOF
}

# ---- 3b. Kernel build toolchain (informational only; never fails the run) ----
check_toolchain() {
  local nv gx
  nv=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' | head -1)
  gx=$(g++ -dumpversion 2>/dev/null | cut -d. -f1)
  echo "       nvcc: ${nv:-not found}   g++: ${gx:-not found}"
  if [ -n "$nv" ] && [ "$(printf '%s\n12.8\n' "$nv" | sort -V | head -1)" != "12.8" ]; then
    echo "       (nvcc < 12.8 cannot build sm_120 kernels; only affects scripts/setup_mamba.sh)"
  fi
  if [ -n "$gx" ] && [ "$gx" -gt 14 ] 2>/dev/null; then
    echo "       (g++ $gx > 14 is rejected by CUDA; setup_mamba.sh installs g++-14/13 automatically)"
  fi
  echo "[PASS] toolchain: recorded (only needed when building mamba-ssm from source)"
  return 0
}
run "env"        check_env
run "gpu-driver" check_gpu_driver
run "python"     check_pyver
run "toolchain"  check_toolchain
run "cuda"       "$PY" scripts/checks/check_cuda.py
run "mamba"      "$PY" scripts/checks/check_mamba.py
run "ollama"     "$PY" scripts/checks/check_ollama.py
run "securebert" "$PY" scripts/checks/check_securebert.py
run "vt"         "$PY" scripts/checks/check_apis.py vt
run "otx"        "$PY" scripts/checks/check_apis.py otx
run "rapiddns"   "$PY" scripts/checks/check_apis.py rapiddns

# ---- summary ----
echo "=============================================="
echo "${B}SMOKE TEST SUMMARY - STAGE 0${N}"
[ ${#PASS[@]} -gt 0 ] && echo "  ${G}PASS (${#PASS[@]})${N}: ${PASS[*]}"
[ ${#FAIL[@]} -gt 0 ] && echo "  ${R}FAIL (${#FAIL[@]})${N}: ${FAIL[*]}"
[ ${#SKIP[@]} -gt 0 ] && echo "  ${Y}SKIP (${#SKIP[@]})${N}: ${SKIP[*]}"
if [ ${#FAIL[@]} -eq 0 ] && [ ${#SKIP[@]} -eq 0 ]; then
  echo "${G}${B}=> Stage 0 acceptance PASSED. Ready for Stage 1.${N}"
  exit 0
else
  echo "${R}${B}=> Stage 0 acceptance NOT met${N} (every check must pass, with no skips)."
  echo "   Read the hint printed under each FAIL, and the Troubleshooting section of README.md."
  exit 1
fi
