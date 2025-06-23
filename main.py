import torch
from transformers import BitsAndBytesConfig, CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig

from datasets import build_dataset
from datasets.utils import DatasetWrapper # Import lớp Wrapper trực tiếp
from torch.utils.data import DataLoader # Import DataLoader của PyTorch

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
    dataset = build_dataset(args.dataset, args.root_path, args.shots, processor.image_processor)
    
    
    train_dataset_wrapper = DatasetWrapper(dataset.train_x, input_size=224, transform=processor.image_processor, is_train=True)
    train_loader = DataLoader(
        train_dataset_wrapper,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2, # Giảm num_workers để tránh warning trên Colab
        pin_memory=True,
        collate_fn=simple_collate_fn # Sử dụng collate_fn của chúng ta
    )

    eval_batch_size = args.batch_size * 2
    val_dataset_wrapper = DatasetWrapper(dataset.val, input_size=224, transform=processor.image_processor, is_train=False)
    val_loader = DataLoader(
        val_dataset_wrapper,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=simple_collate_fn
    )

    test_dataset_wrapper = DatasetWrapper(dataset.test, input_size=224, transform=processor.image_processor, is_train=False)
    test_loader = DataLoader(
        test_dataset_wrapper,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=simple_collate_fn
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