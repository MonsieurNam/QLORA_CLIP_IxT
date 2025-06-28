import torch
from transformers import BitsAndBytesConfig, CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig

from datasets import build_dataset
from datasets.utils import DatasetWrapper # Import lớp Wrapper trực tiếp
from torch.utils.data import DataLoader # Import DataLoader của PyTorch
from torchvision.transforms import (
    Compose, ToTensor, Normalize, RandomResizedCrop, 
    RandomHorizontalFlip, InterpolationMode, RandAugment
)

from run_utils import get_arguments, set_random_seed
from trainer import Trainer

def simple_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images_tensor = torch.stack(images, dim=0)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return images_tensor, targets_tensor

def main():
    args = get_arguments()
    set_random_seed(args.seed)
    
    print("===== Cấu hình Thí nghiệm =====")
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")
    print("==============================\n")

    print(">>> Bước 1: Thiết lập Mô hình QLoRA...")
    model_id_map = {
        'ViT-B/16': 'openai/clip-vit-base-patch16',
        'ViT-B/32': 'openai/clip-vit-base-patch32',
        'ViT-L/14': 'openai/clip-vit-large-patch14',
    }
    model_id = model_id_map.get(args.backbone)
    if model_id is None:
        raise ValueError(f"Backbone '{args.backbone}' không được hỗ trợ trong model_id_map.")

    if args.compute_dtype == 'auto':
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    else:
        compute_dtype = torch.bfloat16 if args.compute_dtype == 'bf16' else torch.float16
    print(f"Sử dụng compute_dtype: {compute_dtype}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Đang tải mô hình '{model_id}' với QLoRA...")
    model = CLIPModel.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    processor = CLIPProcessor.from_pretrained(model_id)

    print("\nÁp dụng cấu hình LoRA (PEFT)...")
    target_modules = [p + "_proj" for p in args.params]
    
    lora_config = LoraConfig(
        r=args.r, lora_alpha=args.alpha, target_modules=target_modules,
        lora_dropout=args.dropout_rate, bias="none", task_type="VISION_TEXT_DUAL_ENCODER"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n>>> Bước 2: Chuẩn bị Dữ liệu...")
    dataset = build_dataset(args.dataset, args.root_path, args.shots)
    
    
    eval_transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Định nghĩa Train Transform với Augmentation mạnh mẽ
    train_transform = Compose([
        RandomResizedCrop(size=224, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(p=0.5),
        RandAugment(num_ops=2, magnitude=9), # Augmentation mạnh
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    train_dataset_wrapper = DatasetWrapper(dataset.train_x, transform=train_transform)
    val_dataset_wrapper = DatasetWrapper(dataset.val, transform=eval_transform)
    test_dataset_wrapper = DatasetWrapper(dataset.test, transform=eval_transform)
    
    train_loader = DataLoader(
        train_dataset_wrapper,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2, 
        pin_memory=True,
        collate_fn=simple_collate_fn 
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
    
    print(f"Dataset: {args.dataset}")
    print(f"  - Số mẫu huấn luyện (train_x): {len(dataset.train_x)}")
    print(f"  - Số mẫu validation (val): {len(dataset.val)}")
    print(f"  - Số mẫu kiểm tra (test): {len(dataset.test)}")
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