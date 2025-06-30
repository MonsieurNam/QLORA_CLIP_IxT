#!/bin/bash

# ======================================================
#            Script chạy thí nghiệm DoRA
# ======================================================
# Chế độ này sử dụng DoRA (Weight-Decomposed Low-Rank Adaptation)
# thay cho LoRA. Nó có thể được kết hợp với cả chế độ 'qlora'
# và 'lora'. Ở đây ta ví dụ với 'qlora'.

# --- Cấu hình Thí nghiệm ---
SHOTS=16
BACKBONE="ViT-B/16"
SEED=42
BATCH_SIZE=8
ACCUM_STEPS=4
LEARNING_RATE=1e-4
RANK=16
ALPHA=32
N_ITERS=$((500 * SHOTS))

# Danh sách các dataset bạn muốn chạy
DATASETS=("food101" "dtd")

# --- Vòng lặp chạy Thí nghiệm ---
for DATASET_NAME in "${DATASETS[@]}"; do

  echo "==========================================================="
  echo "Bắt đầu huấn luyện DoRA (trên QLoRA) trên: ${DATASET_NAME}"
  echo "==========================================================="

  python main.py \
    --mode "qlora" \
    --use_dora \
    --dataset "$DATASET_NAME" \
    --root_path "./data" \
    --shots "$SHOTS" \
    --backbone "$BACKBONE" \
    --lr "$LEARNING_RATE" \
    --r "$RANK" \
    --alpha "$ALPHA" \
    --seed "$SEED" \
    --lora_target_modules q_proj k_proj v_proj out_proj fc1 fc2 \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$ACCUM_STEPS" \
    --n_iters "$N_ITERS" \
    --save_path "./checkpoints"

  echo "Hoàn tất huấn luyện DoRA trên: ${DATASET_NAME}"
  echo ""

done

echo "Tất cả các thí nghiệm DoRA đã hoàn tất!"