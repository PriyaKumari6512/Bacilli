#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Create conda environment for TB Bacilli Segmentation
# =============================================================================
# Usage: bash setup_env.sh
#
# NOTE: mamba-ssm requires Linux + CUDA (sm_70+).
#       On Windows, use WSL2 with an NVIDIA GPU and CUDA toolkit installed.
# =============================================================================

set -euo pipefail

ENV_NAME="tb_bacilli"

echo "============================================"
echo " TB Bacilli Segmentation — Environment Setup"
echo "============================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found. Install Miniconda or Anaconda first."
    exit 1
fi

# Create environment if it doesn't exist
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Environment '${ENV_NAME}' already exists. Activating..."
else
    echo "[INFO] Creating conda environment '${ENV_NAME}' with Python 3.10..."
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "[INFO] Installing PyTorch (CUDA 11.8)..."
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

echo "[INFO] Installing core dependencies..."
pip install \
    numpy>=1.24.0 \
    opencv-python>=4.8.0 \
    Pillow>=10.0.0 \
    scikit-image>=0.21.0 \
    scikit-learn>=1.3.0 \
    scipy>=1.11.0 \
    albumentations>=1.3.1 \
    tensorboard>=2.14.0 \
    tqdm>=4.66.0 \
    matplotlib>=3.7.0

echo "[INFO] Installing causal-conv1d (must be before mamba-ssm)..."
pip install causal-conv1d>=1.1.0

echo "[INFO] Installing mamba-ssm..."
pip install mamba-ssm>=1.1.1

echo "[INFO] Installing optional dependencies..."
pip install \
    timm>=0.9.0 \
    einops>=0.7.0

# Optional: staintools for Macenko normalization
echo "[INFO] Attempting to install staintools (optional)..."
pip install staintools 2>/dev/null || echo "[WARN] staintools install failed — will use pure numpy Macenko fallback."

echo ""
echo "============================================"
echo " Environment '${ENV_NAME}' is ready!"
echo " Activate with: conda activate ${ENV_NAME}"
echo "============================================"
