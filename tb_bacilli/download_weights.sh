#!/usr/bin/env bash
# =============================================================================
# download_weights.sh — Download pretrained weights for CaVMamba encoder
# =============================================================================
# Idempotent: skips download if weights already exist.
# Primary: VMamba-Small from official GitHub releases.
# Fallback: MambaVision-T-1K from HuggingFace (encoder only).
# =============================================================================

set -euo pipefail

VMAMBA_DIR="./pretrained_weights/vmamba"
MAMBAVISION_DIR="./pretrained_weights/mambavision"
VMAMBA_WEIGHT_FILE="${VMAMBA_DIR}/vmamba_small_e238_ema.pth"

echo "============================================"
echo " Pretrained Weight Download"
echo "============================================"

# -------------------------------------------------------------------
# PRIMARY: VMamba-Small
# -------------------------------------------------------------------
if [ -f "${VMAMBA_WEIGHT_FILE}" ]; then
    echo "[INFO] VMamba-Small weights already exist at ${VMAMBA_WEIGHT_FILE}. Skipping."
else
    echo "[INFO] Downloading VMamba-Small pretrained weights..."
    mkdir -p "${VMAMBA_DIR}"

    # Official VMamba GitHub release URL (MzeroMiko/VMamba)
    VMAMBA_URL="https://github.com/MzeroMiko/VMamba/releases/download/v2cls/vssmsmall_dp03_ckpt_epoch_238.pth"

    if wget -q --show-progress -O "${VMAMBA_WEIGHT_FILE}" "${VMAMBA_URL}" 2>/dev/null; then
        echo "[INFO] VMamba-Small weights downloaded successfully."
    elif curl -fSL -o "${VMAMBA_WEIGHT_FILE}" "${VMAMBA_URL}" 2>/dev/null; then
        echo "[INFO] VMamba-Small weights downloaded successfully (via curl)."
    else
        echo "[WARN] VMamba-Small download failed. Attempting fallback..."
        rm -f "${VMAMBA_WEIGHT_FILE}"

        # ---------------------------------------------------------------
        # FALLBACK: MambaVision-T-1K from HuggingFace
        # ---------------------------------------------------------------
        if [ -d "${MAMBAVISION_DIR}" ] && [ "$(ls -A ${MAMBAVISION_DIR})" ]; then
            echo "[INFO] MambaVision weights already exist at ${MAMBAVISION_DIR}. Skipping."
        else
            echo "[INFO] Downloading MambaVision-T-1K encoder weights from HuggingFace..."
            mkdir -p "${MAMBAVISION_DIR}"

            python3 -c "
from transformers import AutoModel
import torch, os

print('[INFO] Loading MambaVision-T-1K from nvidia/MambaVision-T-1K...')
model = AutoModel.from_pretrained('nvidia/MambaVision-T-1K', trust_remote_code=True)

# Extract encoder weights only
encoder_state = {}
for k, v in model.state_dict().items():
    if 'head' not in k.lower() and 'classifier' not in k.lower():
        encoder_state[k] = v

save_path = os.path.join('${MAMBAVISION_DIR}', 'mambavision_t1k_encoder.pth')
torch.save(encoder_state, save_path)
print(f'[INFO] MambaVision encoder weights saved to {save_path}')
print(f'[INFO] Saved {len(encoder_state)} parameter tensors.')
"
            if [ $? -eq 0 ]; then
                echo "[INFO] MambaVision fallback weights downloaded successfully."
            else
                echo "[WARN] Both VMamba and MambaVision downloads failed."
                echo "[WARN] Models will initialize with random weights."
            fi
        fi
    fi
fi

echo ""
echo "============================================"
echo " Weight download complete."
echo "============================================"
