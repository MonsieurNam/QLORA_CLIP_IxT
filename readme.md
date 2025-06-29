
# Project structure
```
qlora-clip/
├── main.py
├── trainer.py
├── run_utils.py
├── utils.py      
├── run_lora.sh
├── run_qlora.sh
├── run_dora.sh
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
└── DATA/
    └── ...
```

# Check VRAM
```
watch -n 1 nvidia-smi
```
# run with RTX4090
```
chmod +x run_qlora.sh
chmod +x run_lora.sh
chmod +x run_dora.sh
./run_qlora.sh
./run_lora.sh
./run_dora.sh
```

# run with Colab gpu T4
```
!chmod +x run_qlora.sh
!chmod +x run_lora.sh
!chmod +x run_dora.sh

!bash /content/QLORA_CLIP_IxT/run_qlora.sh
!bash /content/QLORA_CLIP_IxT/run_lora.sh
!bash /content/QLORA_CLIP_IxT/run_dora.sh
```