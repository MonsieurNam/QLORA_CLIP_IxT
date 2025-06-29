#!/bin/bash

# ======================================================
#            Script chạy thí nghiệm QLoRA
# ======================================================
# Chế độ này lượng tử hóa mô hình base thành 4-bit và
# thường cho kết quả tốt nhất khi áp dụng LoRA cho tất cả
# các lớp linear của Transformer.

# --- Cấu hình Thí nghiệm ---
SHOTS=16
BACKBONE="ViT-B/16"
SEED=42
BATCH_SIZE=8
ACCUM_STEPS=4
LEARNING_RATE=1e-4  # QLoRA thường hoạt động tốt với LR cao hơn một chút
RANK=16
ALPHA=32
N_ITERS=5000

# Danh sách các dataset bạn muốn chạy
# 'dtd' 'eurosat' 'food101' 'oxford_pets' 'stanford_cars' 'ucf101' 'imagenet'
DATASETS=("food101" "dtd")

# --- Vòng lặp chạy Thí nghiệm ---
for DATASET_NAME in "${DATASETS[@]}"; do

  echo "==========================================================="
  echo "Bắt đầu huấn luyện QLoRA trên: ${DATASET_NAME}"
  echo "==========================================================="

  python main.py \
    --mode "qlora" \
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

  echo "Hoàn tất huấn luyện QLoRA trên: ${DATASET_NAME}"
  echo ""

done

echo "Tất cả các thí nghiệm QLoRA đã hoàn tất!"