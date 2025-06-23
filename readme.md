
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
chmod +x run_all.sh
.\run_all.sh
```

# run with Colab gpu T4
```
!chmod +x /content/QLORA_CLIP_IxT/run_all.sh
!bash /content/QLORA_CLIP_IxT/run_all.sh
```