# @title main.py 

from __future__ import annotations

import os
import torch
torch.backends.cudnn.benchmark = False

from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPModel, CLIPProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from peft.utils import prepare_model_for_kbit_training

from datasets import build_dataset
from datasets.utils import DatasetWrapper
from run_utils import get_arguments, set_random_seed, print_gpu_memory_usage, get_system_vram_usage

from torchvision.transforms import (
    Compose, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip,
    Resize, CenterCrop, InterpolationMode,
)
from torch.utils.data import DataLoader

from trainer import Trainer

def prepare_vision_model_for_kbit_training(model, use_gradient_checkpointing=True):
    """
    Hàm chuẩn bị mô hình Vision cho QLoRA, đã được sửa lỗi sụt giảm hiệu suất.
    """
    for name, param in model.named_parameters():
        if param.dtype == torch.uint8:
            param.requires_grad = False

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.to(torch.float32)

    if hasattr(model, 'visual_projection'):
        model.visual_projection.to(torch.float32)
    if hasattr(model, 'text_projection'):
        model.text_projection.to(torch.float32)

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        # Vision embeddings
        if hasattr(model, 'vision_model') and hasattr(model.vision_model, 'embeddings'):
            model.vision_model.embeddings.register_forward_hook(make_inputs_require_grad)
        # Text embeddings
        if hasattr(model, 'text_model') and hasattr(model.text_model, 'embeddings'):
            model.text_model.embeddings.register_forward_hook(make_inputs_require_grad)

    return model

def simple_collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return images, targets


def main():
    args = get_arguments()
    set_random_seed(args.seed)

    os.environ.setdefault("BITSANDBYTES_NOWEIGHT_CACHE", "1")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\n===== Cấu hình Thí nghiệm =====")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("==============================\n")

    model_id = {
        "ViT-B/16": "openai/clip-vit-base-patch16",
        "ViT-B/32": "openai/clip-vit-base-patch32",
        "ViT-L/14": "openai/clip-vit-large-patch14",
        "ViT-bigG-14": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    }.get(args.backbone)
    if model_id is None:
        raise ValueError(f"Backbone '{args.backbone}' không được hỗ trợ.")

    quantization_config = None
    compute_dtype = torch.bfloat16 if args.compute_dtype in ["auto", "bf16"] else torch.float16
    model_kwargs = {}

    if "laion" in model_id:
        print("    -> Phát hiện mô hình LAION OpenCLIP, sẽ sử dụng 'trust_remote_code=True'.")
        model_kwargs["trust_remote_code"] = True

    if args.mode == "qlora":
        print("Chế độ Q-LoRA: Cấu hình BitsAndBytes 4-bit…")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": 0}
    else: # mode == 'lora'
        print("Chế độ LoRA: Mô hình sẽ được tải ở dạng 16-bit.")
        model_kwargs["torch_dtype"] = compute_dtype
        model_kwargs["device_map"] = "auto"

    print(f"  - Tải mô hình '{model_id}' với các tham số: {model_kwargs}")
    model = AutoModel.from_pretrained(model_id, **model_kwargs)
    print("  - Tải Processor...")
    processor = AutoProcessor.from_pretrained(model_id, **({"trust_remote_code": True} if "laion" in model_id else {}))

    # if quantization_config is not None:
    #     model_kwargs["quantization_config"] = quantization_config
    # else:
    #     dtype = torch.bfloat16 if args.compute_dtype in ["auto", "bf16"] else torch.float16
    #     model_kwargs["torch_dtype"] = dtype

    # model: CLIPModel = CLIPModel.from_pretrained(model_id, **model_kwargs)
    # processor = CLIPProcessor.from_pretrained(model_id)

    # if args.mode == "qlora":
    #     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)


    if args.mode == "qlora":
        print(f"  - Chuẩn bị mô hình Vision K-bit training (Gradient Checkpointing: {args.use_gradient_checkpointing})...")
        # Gọi hàm tùy chỉnh mới của chúng ta
        model = prepare_vision_model_for_kbit_training(
            model, use_gradient_checkpointing=args.use_gradient_checkpointing
        )

    print("  - Gắn các adapter LoRA…")

    # Chuẩn bị gradient_checkpointing_kwargs
    grad_ckpt_kwargs = {}
    if args.use_gradient_checkpointing:
        print("  - Cấu hình để bật Gradient Checkpointing trong LoRA...")
        grad_ckpt_kwargs = {"use_reentrant": False}

    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout_rate,
        bias="none",
        target_modules=args.lora_target_modules,
    )

    # Lấy mô hình PEFT và kích hoạt GC cùng lúc
    model = get_peft_model(
        model,
        lora_cfg,
    )
    print("  - Đóng băng các tham số LayerNorm và Bias...")
    frozen_params_count = 0
    for n, p in model.named_parameters():
        if 'ln_' in n or '.bias' in n:
            p.requires_grad_(False)
            frozen_params_count += 1
    print(f"    -> Đã đóng băng {frozen_params_count} tham số.")
    print_gpu_memory_usage("Sau khi tải và chuẩn bị mô hình")

    model.print_trainable_parameters()
    if hasattr(model, 'logit_scale'):
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