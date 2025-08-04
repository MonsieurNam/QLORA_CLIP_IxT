#!/bin/bash
SHOTS=4
BACKBONE="ViT-B/16"
SEED=1
LEARNING_RATE=2e-4
DROPOUT=0.25
RANK=2
ALPHA=1
N_ITERS=500
LORA_B_MULTIPLIER=1.25

export TOKENIZERS_PARALLELISM=false
# Danh sách các dataset bạn muốn chạy
# 'dtd', 'eurosat', 'caltech101', 'food101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 'sun397', 'ucf101', 'imagenet', 'fgvc'

DATASETS=('caltech101')
CONFIGS=(
  "32 1"
  "16 2"
  "8 4"
  "4 8"
  "2 16"
  "1 32"
)
for DATASET_NAME in "${DATASETS[@]}"; do

  echo "==========================================================="
  echo "Bắt đầu huấn luyện QLoRA trên: ${DATASET_NAME}"
  echo "==========================================================="
  for CONFIG in "${CONFIGS[@]}"; do
    BATCH_SIZE=$(echo "$CONFIG" | cut -d' ' -f1)
    ACCUM_STEPS=$(echo "$CONFIG" | cut -d' ' -f2)

    echo "----> Chạy với batch_size=${BATCH_SIZE}, accum_steps=${ACCUM_STEPS}"
    python main.py \
      --mode "qlora" \
      --dataset "$DATASET_NAME" \
      --root_path "/root/DATA" \
      --shots "$SHOTS" \
      --backbone "$BACKBONE" \
      --lr "$LEARNING_RATE" \
      --r "$RANK" \
      --alpha "$ALPHA" \
      --seed "$SEED" \
      --dropout_rate "$DROPOUT" \
      --lora_target_modules q_proj k_proj v_proj\
      --batch_size "$BATCH_SIZE" \
      --gradient_accumulation_steps "$ACCUM_STEPS" \
      --n_iters "$N_ITERS" \
      --save_path "/root/checkpoints/${DATASET_NAME}/bs${BATCH_SIZE}_acc${ACCUM_STEPS}"
    echo "✅ Hoàn tất batch_size=${BATCH_SIZE}, accum_steps=${ACCUM_STEPS}"
    echo ""
  done
  echo "Hoàn tất huấn luyện QLoRA trên: ${DATASET_NAME}"
  echo ""

done

echo "Tất cả các thí nghiệm QLoRA đã hoàn tất!"