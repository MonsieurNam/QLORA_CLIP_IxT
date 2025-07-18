import torch
import math 
import bitsandbytes as bnb
from bitsandbytes.functional import dequantize_4bit
from transformers import BitsAndBytesConfig, CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig
from peft.tuners.lora import LoraLayer

from datasets import build_dataset
from datasets.utils import DatasetWrapper
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, ToTensor, Normalize, RandomResizedCrop,
    RandomHorizontalFlip, InterpolationMode, ColorJitter, transforms, Resize, CenterCrop
)
from run_utils import get_arguments, set_random_seed
from trainer import Trainer

def simple_collate_fn(batch):
    """Collate function đơn giản để tạo batch."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images_tensor = torch.stack(images, dim=0)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return images_tensor, targets_tensor

def main():
    args = get_arguments()
    set_random_seed(args.seed)

    print("\n===== Cấu hình Thí nghiệm =====")
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")
    print("==============================\n")

    print(">>> Bước 1: Thiết lập Mô hình...")
    model_id_map = {
        'ViT-B/16': 'openai/clip-vit-base-patch16',
        'ViT-B/32': 'openai/clip-vit-base-patch32',
        'ViT-L/14': 'openai/clip-vit-large-patch14',
    }
    model_id = model_id_map.get(args.backbone)
    if model_id is None:
        raise ValueError(f"Backbone '{args.backbone}' không được hỗ trợ.")

    if args.compute_dtype == 'auto':
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    else:
        compute_dtype = torch.bfloat16 if args.compute_dtype == 'bf16' else torch.float16

    quantization_config = None
    torch_dtype_for_model = compute_dtype

    if args.mode == 'qlora':
        print(f"Chế độ QLoRA: Đang cấu hình BitsAndBytes với compute_dtype={compute_dtype}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        torch_dtype_for_model = None
    else:
        print(f"Chế độ LoRA gốc: Sử dụng dtype={compute_dtype} cho mô hình.")

    model = CLIPModel.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype_for_model,
        device_map="auto"
    )
    processor = CLIPProcessor.from_pretrained(model_id)

    if args.mode == 'qlora':
        all_possible_linear_layers = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
        modules_to_dequantize = set(all_possible_linear_layers) - set(args.lora_target_modules)

        if modules_to_dequantize:
            print(f"\n>>> Can thiệp thủ công: De-quantize các lớp không được target bởi LoRA...")
            print(f"    Các lớp cần de-quantize: {list(modules_to_dequantize)}")

            high_precision_dtype = model.vision_model.embeddings.patch_embedding.weight.dtype
            print(f"    Sử dụng dtype '{high_precision_dtype}' cho các lớp de-quantized.")

            for name, module in model.named_modules():
                module_name_part = name.split('.')[-1]

                if module_name_part in modules_to_dequantize and isinstance(module, bnb.nn.Linear4bit):
                    parent_name, child_name = name.rsplit('.', 1)
                    parent_module = model.get_submodule(parent_name)

                    print(f"      - Đang xử lý: {name}")

                    new_linear_module = torch.nn.Linear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        device=module.weight.device,
                        dtype=high_precision_dtype
                    )

                    dequantized_weight = dequantize_4bit(
                        module.weight.data, module.weight.quant_state
                    ).to(high_precision_dtype)

                    new_linear_module.weight.data.copy_(dequantized_weight)

                    if module.bias is not None:
                        new_linear_module.bias.data.copy_(module.bias.to(high_precision_dtype))

                    setattr(parent_module, child_name, new_linear_module)
            print("    Hoàn tất de-quantize thủ công.\n")

    print(f"Áp dụng cấu hình PEFT cho các module: {args.lora_target_modules}")
    peft_config_params = {
        "r": args.r,
        "lora_alpha": args.alpha,
        "target_modules": args.lora_target_modules,
        "lora_dropout": args.dropout_rate,
        "bias": "none",
        "task_type": "VISION_TEXT_DUAL_ENCODER"
    }
    if args.use_dora:
        print("DoRA được kích hoạt.")
        peft_config_params["use_dora"] = True
    peft_config = LoraConfig(**peft_config_params)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if args.r > 0:
        print("\n!!! CẢNH BÁO: Đang can thiệp và thay đổi công thức scaling của PEFT !!!")
        new_scaling = args.alpha / math.sqrt(args.r)
        print(f"Công thức scaling mặc định của PEFT (alpha/r): {args.alpha / args.r}")
        print(f"Công thức scaling mới (alpha/sqrt(r)): {new_scaling:.4f}")
        for module in model.modules():
            if isinstance(module, LoraLayer):
                if 'default' in module.scaling:
                    module.scaling['default'] = new_scaling

    print("\n>>> Bước 2: Chuẩn bị Dữ liệu...")
    dataset = build_dataset(args.dataset, args.root_path, args.shots)

    eval_transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    train_transform = Compose([
        RandomResizedCrop(size=224, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    print("\nSử dụng Data Augmentation giống hệt Dự án 1 (scale 0.08-1.0, không có ColorJitter).")

    train_dataset_wrapper = DatasetWrapper(dataset.train_x, transform=train_transform)
    val_dataset_wrapper = DatasetWrapper(dataset.val, transform=eval_transform)
    test_dataset_wrapper = DatasetWrapper(dataset.test, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset_wrapper, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=simple_collate_fn
    )
    eval_batch_size = args.batch_size * 2
    val_loader = DataLoader(
        val_dataset_wrapper, batch_size=eval_batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=simple_collate_fn
    )
    test_loader = DataLoader(
        test_dataset_wrapper, batch_size=eval_batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=simple_collate_fn
    )

    print(f"\nDataset: {args.dataset} ({args.shots} shots)")
    print(f"  - Số mẫu huấn luyện: {len(dataset.train_x)}")
    print(f"  - Số mẫu validation: {len(dataset.val)}")
    print(f"  - Số mẫu kiểm tra: {len(dataset.test)}")
    print(f"  - Số lớp: {dataset.num_classes}")

    print("\n>>> Bước 3: Khởi tạo và bắt đầu quá trình...")
    trainer = Trainer(args, model, processor, dataset, train_loader, val_loader, test_loader)
    if args.eval_only:
        trainer.evaluate("test")
    else:
        trainer.train()

    print("\nHoàn tất!")

if __name__ == '__main__':
    main()