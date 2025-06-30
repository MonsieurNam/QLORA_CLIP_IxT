#!/bin/bash

# ======================================================
#       Script chạy thí nghiệm LoRA gốc (Classic)
# ======================================================
# Chế độ này không lượng tử hóa mô hình base (dùng fp16/bf16)
# và thường chỉ áp dụng LoRA cho các ma trận attention
# (query, key, value) với rank thấp.

# --- Cấu hình Thí nghiệm ---
SHOTS=16
BACKBONE="ViT-B/16"
SEED=42
BATCH_SIZE=32       # Có thể dùng batch size lớn hơn do không có overhead từ quantization
ACCUM_STEPS=1
LEARNING_RATE=2e-4  # LR theo paper CLIP-LoRA
RANK=2              # Rank thấp theo paper CLIP-LoRA
ALPHA=2             # Alpha thấp theo paper CLIP-LoRA
N_ITERS=$((500 * SHOTS))

# Danh sách các dataset bạn muốn chạy
DATASETS=("food101" "dtd")

# --- Vòng lặp chạy Thí nghiệm ---
for DATASET_NAME in "${DATASETS[@]}"; do

  echo "==========================================================="
  echo "Bắt đầu huấn luyện LoRA gốc trên: ${DATASET_NAME}"
  echo "==========================================================="

  python main.py \
    --mode "lora" \
    --dataset "$DATASET_NAME" \
    --root_path "./data" \
    --shots "$SHOTS" \
    --backbone "$BACKBONE" \
    --lr "$LEARNING_RATE" \
    --r "$RANK" \
    --alpha "$ALPHA" \
    --seed "$SEED" \
    --lora_target_modules q_proj k_proj v_proj \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$ACCUM_STEPS" \
    --n_iters "$N_ITERS" \
    --save_path "./checkpoints"

  echo "Hoàn tất huấn luyện LoRA gốc trên: ${DATASET_NAME}"
  echo ""

done

echo "Tất cả các thí nghiệm LoRA gốc đã hoàn tất!"
