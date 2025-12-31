# Dissecting QLoRA's Efficiency for VLM Adaptation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PEFT%20%7C%20BitsAndBytes-orange)](https://huggingface.co/)

Official implementation for the paper **"Dissecting QLoRA's Efficiency for VLM Adaptation: A Practical Analysis of Static vs. Dynamic Memory"**.

This repository provides a systematic investigation into the practical trade-offs of applying Quantized LoRA (QLoRA) to CLIP-style Vision-Language Models. While QLoRA is a standard for LLMs, our analysis reveals a critical distinction between static and dynamic memory consumption when applied to VLMs.

## 📄 Abstract & Key Findings

While QLoRA is often touted for efficiency, our in-depth analysis on CLIP models reveals a nuanced reality:

* **The Paradox:** While QLoRA reduces *static* model weight memory by **62%**, its *dynamic* memory consumption during training can paradoxically exceed that of standard LoRA due to de-quantization overhead.
* **The Trade-off:** Standard LoRA remains superior in terms of **Accuracy** (+2.48% vs QLoRA) and **Training Speed** (1.8x faster).
* **The Optimization Pathway:** We demonstrate that combining QLoRA with **Gradient Checkpointing** is essential to unlock its true potential, reducing peak training VRAM to an exceptionally low **0.17 GB**, making it the ultimate choice for consumer-grade hardware constraints.

## 📊 Experimental Results

### 1. Resource Consumption Analysis (Memory & Time)

*Benchmarks conducted on NVIDIA RTX 4090.*

| Method | Configuration | Static VRAM (Load) | **Peak VRAM (Training)** | Training Time (Example) |
| :--- | :--- | :--- | :--- | :--- |
| **LoRA** (Baseline) | Balanced (4x8) | 0.29 GB | **0.71 GB** | **06:46** (Fastest) |
| **QLoRA** (Standard) | Balanced (4x8) | **0.11 GB** | 1.02 GB | 12:13 |
| **QLoRA + Grad Checkpointing** | Balanced (4x8) | **0.11 GB** | **0.17 GB** (Lowest) | 14:55 |

> **Insight:** QLoRA saves static memory but incurs higher dynamic memory costs. Only by enabling Gradient Checkpointing do we achieve the lowest possible memory footprint (0.17 GB).

### 2. Performance Benchmark (4-shot Accuracy)

Comparison of Average Accuracy across 8 datasets (EuroSAT, DTD, Caltech101, OxfordPets, OxfordFlowers, Food101, FGVC, UCF101).

| Method | Average Accuracy | Notes |
| :--- | :--- | :--- |
| **CLIP-LoRA (Zanella et al.)** | **78.79%** | Best Accuracy & Speed |
| **CLIP-QLoRA (Ours)** | 76.31% | Best Memory Efficiency (with GC) |

While 4-bit quantization incurs a slight performance cost (~2.5%), it remains a competitive option for extreme hardware constraints.

## 🛠 Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MonsieurNam/QLORA_CLIP_IxT.git](https://github.com/MonsieurNam/QLORA_CLIP_IxT.git)
    cd QLORA_CLIP_IxT
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch==2.1.1 torchvision==0.16.1
    pip install -r requirements.txt
    ```

## 🚀 Usage

We provide shell scripts to reproduce our experiments. You can switch between LoRA and QLoRA modes.

### Training Command

To reproduce the **low-memory QLoRA** results (0.17 GB VRAM), ensure you enable gradient checkpointing or use a high accumulation step count if mimicking the "Low Memory" setup from the paper.

```bash
# Example: Run QLoRA with high memory efficiency
python3 main.py \
    --mode "qlora" \
    --dataset "caltech101" \
    --shots 4 \
    --batch_size 4 \
    --gradient_accumulation_steps 8
```
### To run the full benchmark suite:

```bash
chmod +x run.sh
./run.sh
```

## 📂 Project Structure
```bash
📦QLORA_CLIP_IxT
 ┣ 📂datasets          # Data loaders for 8 benchmark datasets
 ┣ 📂logs_full_dataset # Training logs
 ┣ 📜main.py           # Main entry point
 ┣ 📜trainer.py        # Training logic (LoRA/QLoRA implementation)
 ┣ 📜run.sh            # Experiment script
 ┗ 📜requirements.txt
```

## 🎓 Citation
### If you use this code or findings in your research, please cite our paper:

```bash

@article{nam2025dissecting,
  title={Dissecting QLoRA's Efficiency for VLM Adaptation: A Practical Analysis of Static vs. Dynamic Memory},
  author={Nguyen, Ngo Nhat Nam and Le, Phan Quynh Nhi and Nguyen, Vinh Nghi and Le, Ngoc Anh Thu and Tran, Ngoc Hoang},
  journal={FPT University Can Tho},
  year={2025}
}
```

### 👥 Authors & Acknowledgements

Authors: Nguyen Ngo Nhat Nam, Le Phan Quynh Nhi, Nguyen Vinh Nghi, Le Ngoc Anh Thu, Tran Ngoc Hoang. Affiliation: Department of Artificial Intelligence, FPT University Can Tho, Vietnam.

This work builds upon the CLIP-LORA baseline. We thank the open-source community for libraries like bitsandbytes and peft.







