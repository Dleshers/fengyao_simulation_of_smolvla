#!/usr/bin/env bash
set -e

export SIMULATION_STORAGE_ROOT="${SIMULATION_STORAGE_ROOT:-/media/radu/Storage/SmolVLA-Fengyao/simulation_env}"
export SIMULATION_FAST_ROOT="${SIMULATION_FAST_ROOT:-/home/radu/.local/share/smolvla_sim_runtime}"

requested_env="${1:-smolvla}"
if [[ -x "$SIMULATION_FAST_ROOT/conda/envs/$requested_env/bin/python" ]]; then
    runtime_root="$SIMULATION_FAST_ROOT"
else
    runtime_root="$SIMULATION_STORAGE_ROOT"
fi

export CONDA_PKGS_DIRS="$runtime_root/cache/conda_pkgs"
export PIP_CACHE_DIR="$runtime_root/cache/pip"
export HF_HOME="${HF_HOME:-$SIMULATION_FAST_ROOT/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$SIMULATION_FAST_ROOT/cache/torch}"
export OMNI_USER_DIR="$runtime_root/cache/ov"
export WANDB_DIR="${WANDB_DIR:-$SIMULATION_FAST_ROOT/cache/wandb}"

if [[ "$requested_env" == "isaaclab" ]]; then
    env_prefix="$runtime_root/conda/envs/isaaclab"
else
    env_prefix="$runtime_root/conda/envs/smolvla"
fi

source /home/radu/miniconda3/etc/profile.d/conda.sh
conda activate "$env_prefix"
if [[ "$requested_env" == "isaaclab" && -f "$runtime_root/isaac-sim-4.5.0/setup_conda_env.sh" ]]; then
    source "$runtime_root/isaac-sim-4.5.0/setup_conda_env.sh"
    # This host has duplicate NVIDIA ICD manifests. Select one per process so
    # Kit does not enumerate the same RTX 4090 twice.
    export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
    export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
fi
export PATH="$CONDA_PREFIX/bin:/usr/bin:/bin:$PATH"
if cmake_native_bin="$(python -c 'import cmake; print(cmake.CMAKE_BIN_DIR)' 2>/dev/null)"; then
    export PATH="$cmake_native_bin:$PATH"
fi
unset cmake_native_bin

if [[ "$requested_env" != "isaaclab" ]]; then
    export PYTHONPATH="$runtime_root/lerobot-tactile/src${PYTHONPATH:+:$PYTHONPATH}"
fi
echo "Activated: $env_prefix"
unset runtime_root
unset requested_env
