SHOTS=1
BACKBONE="ViT-B/16"
SEED=1
BATCH_SIZE=32       
ACCUM_STEPS=1
LEARNING_RATE=2e-4  
RANK=2             
ALPHA=16             
N_ITERS=$((500 * SHOTS))
DROPOUT=0.25
# 'dtd', 'eurosat', 'caltech101', 'food101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 'sun397', 'ucf101', 'imagenet', 'fgvc'

DATASETS=("oxford_flowers")

for DATASET_NAME in "${DATASETS[@]}"; do

  echo "==========================================================="
  echo "Bắt đầu huấn luyện LoRA gốc trên: ${DATASET_NAME}"
  echo "==========================================================="

  python main.py \
    --mode "lora" \
    --dataset "$DATASET_NAME" \
    --root_path "/content/DATA" \
    --shots "$SHOTS" \
    --backbone "$BACKBONE" \
    --lr "$LEARNING_RATE" \
    --r "$RANK" \
    --alpha "$ALPHA" \
    --dropout_rate "$DROPOUT" \
    --seed "$SEED" \
    --lora_target_modules q_proj k_proj v_proj \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$ACCUM_STEPS" \
    --n_iters "$N_ITERS" \
    --save_path "/content/checkpoints"

  echo "Hoàn tất huấn luyện LoRA gốc trên: ${DATASET_NAME}"
  echo ""

done

echo "Tất cả các thí nghiệm LoRA gốc đã hoàn tất!"
