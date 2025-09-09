#!/bin/bash
SHOTS_DEFAULT=1
BACKBONE="ViT-B/16"
SEED=1
LEARNING_RATE=2e-4
DROPOUT=0.25
RANK=2
ALPHA=1
N_ITERS=500
export TOKENIZERS_PARALLELISM=false

MODE="lora"

DATASETS=('dtd' 'eurosat' 'caltech101' 'oxford_pets' 'oxford_flowers' 'ucf101' 'fgvc')

BS_ACC_PAIRS=("1:32" "2:16" "4:8" "8:4" "16:2" "32:1")

SHOTS_LIST=(1 4 16)

LOG_DIR="./logs/${MODE}"
mkdir -p "$LOG_DIR"

for SHOTS in "${SHOTS_LIST[@]}"; do

  for PAIR in "${BS_ACC_PAIRS[@]}"; do
    IFS=":" read -r BATCH_SIZE ACCUM_STEPS <<< "$PAIR"

    for DATASET_NAME in "${DATASETS[@]}"; do
      echo "==========================================================="
      echo "Bắt đầu huấn luyện ${MODE} | dataset=${DATASET_NAME} | shots=${SHOTS} | bs=${BATCH_SIZE} | acc=${ACCUM_STEPS}"
      echo "==========================================================="

      LOG_FILE="${LOG_DIR}/${DATASET_NAME}_shots${SHOTS}_bs${BATCH_SIZE}_acc${ACCUM_STEPS}_${MODE}.log"

      python3 main.py \
        --mode "$MODE" \
        --dataset "$DATASET_NAME" \
        --root_path "/root/DATA" \
        --shots "$SHOTS" \
        --backbone "$BACKBONE" \
        --lr "$LEARNING_RATE" \
        --r "$RANK" \
        --alpha "$ALPHA" \
        --seed "$SEED" \
        --dropout_rate "$DROPOUT" \
        --lora_target_modules q_proj k_proj v_proj \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$ACCUM_STEPS" \
        --n_iters "$N_ITERS" \
        --save_path "/root/checkpoints" \
        >"$LOG_FILE" 2>&1

      echo "Hoàn tất ${MODE} trên: ${DATASET_NAME} (shots=${SHOTS}, bs=${BATCH_SIZE}, acc=${ACCUM_STEPS})"
      echo "Log: $LOG_FILE"
      echo ""
    done
  done
done

echo "Tất cả các thí nghiệm ${MODE} đã hoàn tất!"
