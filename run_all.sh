SHOTS=16
BACKBONE="ViT-B/16"
SEED=1
BATCH_SIZE=8
ACCUM_STEPS=4
LEARNING_RATE=2e-5
RANK=16
ALPHA=32

DATASETS=("food101") # Thêm imagenet nếu bạn muốn
# 'dtd', 'eurosat', 'caltech101', 'food101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 'sun397', 'ucf101', 'imagenet', 'fgvc'

for DATASET_NAME in "${DATASETS[@]}"; do

  echo "==========================================================="
  echo "Bắt đầu huấn luyện trên: ${DATASET_NAME}"
  echo "==========================================================="

  SAVE_PATH="checkpoints/${DATASET_NAME}_${SHOTS}shots"

  python main.py \
      --dataset "$DATASET_NAME" \
      --root_path "/content/DATA" \
      --shots "$SHOTS" \
      --backbone "$BACKBONE" \
      --lr "$LEARNING_RATE"\
      --r "$RANK" \
      --alpha "$ALPHA" \
      --seed "$SEED" \
      --params q k v o \
      --batch_size "$BATCH_SIZE" \
      --gradient_accumulation_steps "$ACCUM_STEPS" \
      --save_path "$SAVE_PATH" \

  echo "Hoàn tất huấn luyện trên: ${DATASET_NAME}"
  echo ""

done

echo "Tất cả các thí nghiệm đã hoàn tất!"