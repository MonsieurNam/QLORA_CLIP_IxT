# %%writefile /content/QLORA_CLIP_IxT/run_all.sh

# --- Cấu hình Thí nghiệm ---
SHOTS=1
BACKBONE="ViT-B/16"
SEED=1
BATCH_SIZE=4
ACCUM_STEPS=8

# Danh sách các bộ dữ liệu bạn muốn chạy
DATASETS=("caltech101") # Thêm imagenet nếu bạn muốn
# "caltech101" "food101" "oxford_flowers" "oxford_pets" "stanford_cars"

# Vòng lặp để chạy qua từng bộ dữ liệu
for DATASET_NAME in "${DATASETS[@]}"; do
  
  echo "==========================================================="
  echo "Bắt đầu huấn luyện trên: ${DATASET_NAME}"
  echo "==========================================================="
  
  # Tạo đường dẫn để lưu adapter cho từng dataset
  SAVE_PATH="checkpoints/${DATASET_NAME}_${SHOTS}shots"
  
  python main.py \
      --dataset "$DATASET_NAME" \
      --root_path "/content/DATA" \
      --shots "$SHOTS" \
      --backbone "$BACKBONE" \
      --seed "$SEED" \
      --batch_size "$BATCH_SIZE" \
      --gradient_accumulation_steps "$ACCUM_STEPS" \
      --save_path "$SAVE_PATH" \
      # Thêm các tham số khác nếu cần, ví dụ: --compute_dtype auto

  echo "Hoàn tất huấn luyện trên: ${DATASET_NAME}"
  echo ""
  
done

echo "Tất cả các thí nghiệm đã hoàn tất!"