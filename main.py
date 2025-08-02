# @title main.py – Đã tích hợp torch.compile() và đóng băng LayerNorm/Bias

from __future__ import annotations

import os
import torch
from transformers import CLIPModel, CLIPProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from peft.utils import prepare_model_for_kbit_training

from datasets import build_dataset
from datasets.utils import DatasetWrapper
from torchvision.transforms import (
    Compose, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip,
    Resize, CenterCrop, InterpolationMode,
)
from torch.utils.data import DataLoader

from run_utils import get_arguments, set_random_seed
from trainer import Trainer

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def simple_collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return images, targets

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = get_arguments()
    set_random_seed(args.seed)

    os.environ.setdefault("BITSANDBYTES_NOWEIGHT_CACHE", "1")

    print("\n===== Cấu hình Thí nghiệm =====")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("==============================\n")

    model_id = {
        "ViT-B/16": "openai/clip-vit-base-patch16",
        "ViT-B/32": "openai/clip-vit-base-patch32",
        "ViT-L/14": "openai/clip-vit-large-patch14",
    }.get(args.backbone)
    if model_id is None:
        raise ValueError(f"Backbone '{args.backbone}' không được hỗ trợ.")

    quantization_config = None
    if args.mode == "qlora":
        print("Chế độ Q-LoRA: Cấu hình BitsAndBytes 4-bit…")
        compute_dtype = (
            torch.bfloat16 if args.compute_dtype in ["auto", "bf16"] else torch.float16
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    print(f"  - Tải mô hình '{model_id}'…")
    model_kwargs = {}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        dtype = torch.bfloat16 if args.compute_dtype in ["auto", "bf16"] else torch.float16
        model_kwargs["torch_dtype"] = dtype

    model: CLIPModel = CLIPModel.from_pretrained(model_id, **model_kwargs).to("cuda")  
    processor = CLIPProcessor.from_pretrained(model_id)

    if args.mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    print("  - Gắn các adapter LoRA…")
    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout_rate,
        bias="none",
        target_modules=args.lora_target_modules,
        task_type="VISION_TEXT_DUAL_ENCODER",
    )
    model = get_peft_model(model, lora_cfg)

    # ==================== THAY ĐỔI 1: ĐÓNG BĂNG LAYER NORM & BIAS ====================
    print("  - Đóng băng các tham số LayerNorm và Bias...")
    frozen_params_count = 0
    for n, p in model.named_parameters():
        if 'ln_' in n or '.bias' in n:
            p.requires_grad_(False)
            frozen_params_count += 1
    print(f"    -> Đã đóng băng {frozen_params_count} tham số.")
    # ==============================================================================

    model.print_trainable_parameters()
    model.logit_scale.requires_grad_(False)

    # Chỉ áp dụng khi phiên bản PyTorch đủ mới
    # if hasattr(torch, '__version__') and torch.__version__ >= "2.0":
    #     print("\n>>> Áp dụng torch.compile() để tối ưu hóa tốc độ...")
    #     model = torch.compile(model, mode="reduce-overhead")
    #     print("    -> Hoàn tất biên dịch mô hình.")
    # else:
    #     print("\n>>> Phiên bản PyTorch < 2.0, bỏ qua torch.compile().")

    print("\n>>> Bước 2: Chuẩn bị Dữ liệu…")
    dataset = build_dataset(args.dataset, args.root_path, args.shots)

    eval_transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224), ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    train_transform = Compose([
        RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC), RandomHorizontalFlip(), ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    train_ds = DatasetWrapper(dataset.train_x, transform=train_transform)
    val_ds = DatasetWrapper(dataset.val, transform=eval_transform)
    test_ds = DatasetWrapper(dataset.test, transform=eval_transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=simple_collate_fn,
    )
    eval_bs = args.batch_size
    val_loader = DataLoader(
        val_ds, batch_size=eval_bs, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=simple_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_bs, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=simple_collate_fn,
    )

    print(f"\nDataset: {args.dataset} ({args.shots} shots)")
    print(f"  - Train: {len(dataset.train_x)} | Val: {len(dataset.val)} | Test: {len(dataset.test)}")
    print(f"  - Classes: {dataset.num_classes}")

    trainer = Trainer(args, model, processor, dataset, train_loader, val_loader, test_loader)
    if args.eval_only:
        trainer.evaluate("test")
    else:
        trainer.train()

    print("\nHoàn tất!")

if __name__ == "__main__":
    main()