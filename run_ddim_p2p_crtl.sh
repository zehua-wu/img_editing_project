#!/bin/bash
set -euo pipefail

# 你要扫的 control_type 列表
CONTROL_TYPES=(
  "normal"
  "depth"
  "pose"
  "canny"
)

# 统一用的 control_scale
CONTROL_SCALE=0.4

# 输出总目录（里面再按 control_type 分子目录）
BASE_OUTPUT="DDIM_p2p_control_output"

# 如果你只想用某块卡，可以在这里指定
# export CUDA_VISIBLE_DEVICES=0

for CT in "${CONTROL_TYPES[@]}"; do
  OUT_DIR="${BASE_OUTPUT}/${CT}_scale${CONTROL_SCALE}"

  echo "==============================="
  echo "Running control_type=${CT}, control_scale=${CONTROL_SCALE}"
  echo "Output dir: ${OUT_DIR}"
  echo "==============================="

  mkdir -p "${OUT_DIR}"

  python run_editing_p2p.py \
    --edit_method_list ddim+p2p_ctrl \
    --control_type "${CT}" \
    --control_scale "${CONTROL_SCALE}" \
    --output_path "${OUT_DIR}"

  echo "Finished control_type=${CT}"
  echo
done

echo "All control types done."
