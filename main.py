# main.py
import torch
from transformers import BitsAndBytesConfig, CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig

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
    # Bước 0: Lấy và thiết lập cấu hình
    args = get_arguments()
    set_random_seed(args.seed)

    print("\n===== Cấu hình Thí nghiệm =====")
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")
    print("==============================\n")

    # Bước 1: Thiết lập Mô hình, Lượng tử hóa và PEFT
    print(">>> Bước 1: Thiết lập Mô hình...")
    model_id_map = {
        'ViT-B/16': 'openai/clip-vit-base-patch16',
        'ViT-B/32': 'openai/clip-vit-base-patch32',
        'ViT-L/14': 'openai/clip-vit-large-patch14',
    }
    model_id = model_id_map.get(args.backbone)
    if model_id is None:
        raise ValueError(f"Backbone '{args.backbone}' không được hỗ trợ.")

    # Cấu hình dtype và quantization dựa trên --mode
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
        torch_dtype_for_model = None  # Không cần thiết khi dùng quantization
    else: # mode == 'lora'
        print(f"Chế độ LoRA gốc: Sử dụng dtype={compute_dtype} cho mô hình.")

    # Tải mô hình với cấu hình phù hợp
    model = CLIPModel.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype_for_model,
        device_map="auto"
    )
    processor = CLIPProcessor.from_pretrained(model_id)

    # Áp dụng cấu hình PEFT (LoRA/DoRA)
    print(f"\nÁp dụng cấu hình PEFT cho các module: {args.lora_target_modules}")
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

    # Bước 2: Chuẩn bị Dữ liệu
    print("\n>>> Bước 2: Chuẩn bị Dữ liệu...")
    dataset = build_dataset(args.dataset, args.root_path, args.shots)

    eval_transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    train_transform = Compose([
        RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

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

    print(f"Dataset: {args.dataset} ({args.shots} shots)")
    print(f"  - Số mẫu huấn luyện: {len(dataset.train_x)}")
    print(f"  - Số mẫu validation: {len(dataset.val)}")
    print(f"  - Số mẫu kiểm tra: {len(dataset.test)}")
    print(f"  - Số lớp: {dataset.num_classes}")

    # Bước 3: Khởi tạo và bắt đầu quá trình
    print("\n>>> Bước 3: Khởi tạo và bắt đầu quá trình...")
    trainer = Trainer(args, model, processor, dataset, train_loader, val_loader, test_loader)
    if args.eval_only:
        trainer.evaluate("test")
    else:
        trainer.train()

    print("\nHoàn tất!")

if __name__ == '__main__':
    main()