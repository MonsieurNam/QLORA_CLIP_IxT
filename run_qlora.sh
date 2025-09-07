#!/bin/bash

DATASETS=('dtd' 'eurosat' 'caltech101' 'food101' 'oxford_pets' 'oxford_flowers' 'ucf101' 'fgvc')
SHOTS=(1)
ADAPTER_TYPE="qlora"
LEARNING_RATE="2e-4"
RANK=2
ALPHA=1
LORA_PARAMS="q_proj k_proj v_proj"
N_ITERS=500
BATCH_SIZE=4
ACCUM_STEPS=8
ROOT_PATH="/root/DATA"
SAVE_PATH="/root/QLORA_CLIP_IxT/checkpoints"
LOG_DIR="/root/QLORA_CLIP_IxT/logs_qlora"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$LOG_DIR"

for DATASET in "${DATASETS[@]}"; do
  for SHOT in "${SHOTS[@]}"; do

    CONFIG_NAME="${ADAPTER_TYPE}_${DATASET}_${SHOT}shot_bs${BATCH_SIZE}_acc${ACCUM_STEPS}"
    LOG_FILE="${LOG_DIR}/${CONFIG_NAME}.log"

    rm -f "$LOG_FILE"

    echo "======================================================================"
    echo "Starting run: ${CONFIG_NAME}"
    echo "======================================================================"

    COMMAND="python3 /root/QLORA_CLIP_IxT/main.py \
      --mode ${ADAPTER_TYPE} \
      --dataset ${DATASET} \
      --root_path ${ROOT_PATH} \
      --shots ${SHOT} \
      --n_iters ${N_ITERS} \
      --batch_size ${BATCH_SIZE} \
      --gradient_accumulation_steps ${ACCUM_STEPS} \
      --backbone ViT-B/16 \
      --dropout_rate 0.25 \
      --seed 1 \
      --lr ${LEARNING_RATE} \
      --r ${RANK} \
      --alpha ${ALPHA} \
      --lora_target_modules ${LORA_PARAMS} \
      --save_path ${SAVE_PATH}/${DATASET}/${CONFIG_NAME}"

    echo "Executing command:"
    echo "${COMMAND}"
    echo ""

    eval ${COMMAND} > "${LOG_FILE}" 2>&1
  done
done

echo "✅ All QLoRA experiments completed."
