
# QLoRA-VLM: Efficient Few-Shot Adaptation of Vision-Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PEFT%20%7C%20BitsAndBytes-orange)](https://huggingface.co/)

Official implementation for the paper **"QLoRA-VLM: Efficient Few-Shot Adaptation of Vision-Language Models with Quantized Low-Rank Adapters"**. This repository provides a comprehensive framework for fine-tuning Vision-Language Models (VLMs) like CLIP using modern Parameter-Efficient Fine-Tuning (PEFT) techniques, including LoRA, QLoRA, and DoRA.

This work starts by replicating and modernizing the strong baseline from [CLIP-LORA (Zanella et al., 2024)](https://github.com/MaxZanella/CLIP-LORA), and extends it to investigate the efficacy of 4-bit quantization. Our findings demonstrate that QLoRA is not only exceptionally memory-efficient but also surprisingly faster than its full-precision LoRA counterpart, all while achieving state-of-the-art accuracy in few-shot settings.

## Key Features & Contributions

- **QLoRA for VLMs:** The first, to our knowledge, systematic study and implementation of QLoRA for few-shot VLM adaptation.
- **Transformative Efficiency:** Drastically reduces resource requirements, with over **60% less VRAM** usage and up to **35% faster training times** compared to standard LoRA.
- **State-of-the-Art Performance:** Matches or exceeds the performance of strong LoRA baselines and otCommingsoon
```
*(Note: Replace `xxxx.xxxxx` with the actual arXiv ID once available.)*

## Acknowledgements
This work builds upon the insights and strong baseline established by the [CLIP-LORA](https://github.com/MaxZanella/CLIP-LORA) project. We are also grateful for the powerful open-source libraries from [Hugging Face](https://huggingface.co/) that made this research possible.
```
