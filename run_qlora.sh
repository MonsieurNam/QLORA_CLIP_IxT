SHOTS=1
BACKBONE="ViT-B/16"
SEED=1
BATCH_SIZE=8
ACCUM_STEPS=4
LEARNING_RATE=1e-4  
DROPOUT=0.25
RANK=16
ALPHA=32
N_ITERS=$((500 * SHOTS))
export TOKENIZERS_PARALLELISM=false

# 'dtd', 'eurosat', 'caltech101', 'food101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 'sun397', 'ucf101', 'imagenet', 'fgvc'

DATASETS=("oxford_flowers" )

for DATASET_NAME in "${DATASETS[@]}"; do

  echo "==========================================================="
  echo "Bắt đầu huấn luyện QLoRA trên: ${DATASET_NAME}"
  echo "==========================================================="

  python main.py \
    --mode "qlora" \
    --dataset "$DATASET_NAME" \
    --root_path "/content/DATA" \
    --shots "$SHOTS" \
    --backbone "$BACKBONE" \
    --lr "$LEARNING_RATE" \
    --r "$RANK" \
    --alpha "$ALPHA" \
    --seed "$SEED" \
    --dropout_rate "$DROPOUT" \
    --lora_target_modules q_proj k_proj v_proj out_proj fc1 fc2\
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$ACCUM_STEPS" \
    --n_iters "$N_ITERS" \
    --save_path "/content/checkpoints"

  echo "Hoàn tất huấn luyện QLoRA trên: ${DATASET_NAME}"
  echo ""

done

echo "Tất cả các thí nghiệm QLoRA đã hoàn tất!"