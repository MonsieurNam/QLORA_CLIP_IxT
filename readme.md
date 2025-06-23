
# Project structure
```
qlora-clip/
├── main.py
├── trainer.py
├── run_utils.py
├── utils.py       <-- File này cũng phải nằm trong thư mục gốc
├── datasets/
│   ├── __init__.py
│   ├── caltech101.py
│   ├── dtd.py
│   ├── eurosat.py
│   ├── fgvc.py
│   ├── food101.py
│   ├── imagenet.py
│   ├── oxford_flowers.py
│   ├── oxford_pets.py
│   ├── stanford_cars.py
│   ├── sun397.py
│   ├── ucf101.py
│   └── utils.py   <-- File utils.py của repo gốc nằm ở đây
└── data/
    └── ...
```

# Check VRAM
```
watch -n 1 nvidia-smi
```
# run with RTX4090
```
python main.py \
    --dataset stanford_cars \
    --shots 16 \
    --backbone "ViT-L/14" \
    --seed 1 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --compute_dtype bf16
```

# run with Colab gpu T4
```
python main.py \
    --dataset eurosat \
    --shots 16 \
    --backbone "ViT-B/16" \
    --seed 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 8
```