
# QLoRA-VLM: Efficient Few-Shot Adaptation of Vision-Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PEFT%20%7C%20BitsAndBytes-orange)](https://huggingface.co/)

Official implementation for the paper **"QLoRA-VLM: Efficient Few-Shot Adaptation of Vision-Language Models with Quantized Low-Rank Adapters"**. This repository provides a comprehensive framework for fine-tuning Vision-Language Models (VLMs) like CLIP using modern Parameter-Efficient Fine-Tuning (PEFT) techniques, including LoRA, QLoRA, and DoRA.

This work starts by replicating and modernizing the strong baseline from [CLIP-LORA (Zanella et al., 2024)](https://github.com/MaxZanella/CLIP-LORA), and extends it to investigate the efficacy of 4-bit quantization. Our findings demonstrate that QLoRA is not only exceptionally memory-efficient but also surprisingly faster than its full-precision LoRA counterpart, all while achieving state-of-the-art accuracy in few-shot settings.

## Key Features & Contributions

- **QLoRA for VLMs:** The first, to our knowledge, systematic study and implementation of QLoRA for few-shot VLM adaptation.
- **Transformative Efficiency:** Drastically reduces resource requirements, with over **60% less VRAM** usage and up to **35% faster training times** compared to standard LoRA.
- **State-of-the-Art Performance:** Matches or exceeds the performance of strong LoRA baselines and other prompt-tuning methods across a wide range of benchmarks.
- **Reproducibility & Modernization:** Built on standard, industry-leading libraries like Hugging Face `Transformers`, `PEFT`, and `BitsAndBytes` for robust and reproducible research.
- **Flexible PEFT Framework:** Easily switch between LoRA, QLoRA, and DoRA fine-tuning strategies via simple command-line arguments.

## Highlighted Results

### 1. Performance Benchmark (4-shot)

Our QLoRA approach achieves a higher average accuracy compared to the original CLIP-LoRA and other prominent methods, demonstrating its effectiveness.

| Method | Aircraft | EuroSAT | Food | Pets | Flowers | Caltech | DTD | UCF | **Average** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PLOT++ (ICLR’23) | 35.3 | 83.2 | 86.5 | 92.6 | 92.9 | 95.1 | 62.4 | 79.8 | 78.48 |
| CLIP-LoRA (Zanella et al.)| **37.9** | 84.9 | 82.7 | 91.0 | **93.7** | 95.2 | 63.8 | **81.1** | 78.79 |
| **CLIP-QLoRA (Ours)** | 34.68 | **89.43** | **84.3** | **92.34** | 91.64 | **95.38** | **66.31** | 80.52 | **79.33** |

### 2. Resource Efficiency on Caltech101

QLoRA provides immense efficiency gains in both memory and speed without compromising accuracy.

| Shots | Method | Training Time | Peak VRAM (GB) | Final Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- |
| **1-shot** | LoRA | 00:04:58 | 4.71 | 93.70 |
| | **QLoRA** | **00:03:08** | **1.75** | **93.79** |
| **4-shot** | LoRA | 00:19:05 | 4.71 | 95.20 |
| | **QLoRA** | **00:12:22** | **1.75** | **95.38** |
| **16-shot**| LoRA | 01:06:36 | 4.71 | 96.40 |
| | **QLoRA** | **00:48:46** | **1.75** | **96.43** |

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MonsieurNam/QLORA_CLIP_IxT.git
    cd /root/QLORA_CLIP_IxT
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    apt install python3.10-venv
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    We recommend using PyTorch 2.1 or higher with CUDA 11.8 or 12.1.
    ```bash
    pip install "numpy<2.0"
    pip install torch==2.1.1 torchvision==0.16.1
    pip install -r requirements.txt
    ```
    
4.  **Download Datasets:**
    The framework supports multiple datasets. You will need to download them and place them in the `DATA/` directory. Refer to `datasets/DATASETS.md` for download links and instructions.

## Training and Evaluation

We provide convenient shell scripts to run experiments for LoRA, QLoRA, and DoRA. You can customize these scripts to change datasets, hyperparameters, and other settings.

### Running on a Local Machine (e.g., with RTX 4090)

1.  **Make scripts executable:**
    ```bash
    chmod +x run_lora.sh run_qlora.sh run_dora.sh
    ```

2.  **Execute the desired script:**
    ```bash
    # To run the main QLoRA experiment
    ./run_qlora.sh

    # To run the full-precision LoRA baseline
    ./run_lora.sh
    
    # To run the DoRA experiment
    ./run_dora.sh
    ```

### Monitoring VRAM Usage

To monitor GPU memory usage during training, open a new terminal and run:
```bash
watch -n 1 nvidia-smi
```

### Running on Google Colab (e.g., with T4 GPU)

1.  **Upload the project** to your Google Drive or clone it directly.
2.  **Run the following commands** in a Colab notebook cell:
    ```python
    # Make scripts executable
    !chmod +x /content/QLORA_CLIP_IxT/run_qlora.sh
    !chmod +x /content/QLORA_CLIP_IxT/run_lora.sh
    !chmod +x /content/QLORA_CLIP_IxT/run_dora.sh

    # Run the QLoRA experiment
    !bash /content/QLORA_CLIP_IxT/run_qlora.sh
    ```

## Project Structure
```
qlora-clip/
├── main.py             # Main entry point for experiments
├── trainer.py          # Core training and evaluation logic
├── run_utils.py        # Argument parsing and setup utilities
├── metrics.py          # Accuracy calculation functions
├── run_lora.sh         # Script to run LoRA experiments
├── run_qlora.sh        # Script to run QLoRA experiments
├── run_dora.sh         # Script to run DoRA experiments
├── datasets/           # Folder containing data loading logic for each dataset
│   ├── __init__.py     # Dataset factory (build_dataset)
│   ├── caltech101.py
│   └── ...
└── DATA/               # Directory to store raw dataset files
    └── ...
```

## Citation
If you find this work useful for your research, please consider citing our paper:

```bibtex
comming soon
```


## Acknowledgements
This work builds upon the insights and strong baseline established by the [CLIP-LORA](https://github.com/MaxZanella/CLIP-LORA) project. We are also grateful for the powerful open-source libraries from [Hugging Face](https://huggingface.co/) that made this research possible.
```
