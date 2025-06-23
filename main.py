import torch
from transformers import BitsAndBytesConfig, CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig

from datasets import build_dataset
from datasets.utils import build_data_loader 

from run_utils import get_arguments, set_random_seed
from trainer import Trainer
from metrics import cls_acc 

def main():
    """
    Hàm chính điều phối toàn bộ quy trình:
    1. Đọc cấu hình.
    2. Thiết lập mô hình QLoRA.
    3. Chuẩn bị dữ liệu.
    4. Khởi tạo và chạy Trainer.
    """
    
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
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            print("Phát hiện hỗ trợ bfloat16. Sử dụng torch.bfloat16.")
        else:
            compute_dtype = torch.float16
            print("Không phát hiện hỗ trợ bfloat16. Sử dụng torch.float16.")
    else:
        compute_dtype = torch.bfloat16 if args.compute_dtype == 'bf16' else torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Đang tải mô hình '{model_id}' với QLoRA...")
    model = CLIPModel.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"  # Tự động phân bổ các lớp lên GPU/CPU nếu cần
    )
    processor = CLIPProcessor.from_pretrained(model_id)

    print("\nÁp dụng cấu hình LoRA (PEFT)...")
    target_modules = []
    if 'q' in args.params: target_modules.append("q_proj")
    if 'v' in args.params: target_modules.append("v_proj")
    if 'k' in args.params: target_modules.append("k_proj")
    if 'o' in args.params: target_modules.append("out_proj")
    
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=target_modules,
        lora_dropout=args.dropout_rate,
        bias="none",
        task_type="VISION_TEXT_DUAL_ENCODER"  # Quan trọng, chỉ định cho các mô hình 2 nhánh
    )
    
    model = get_peft_model(model, lora_config)
    
    print("\nThông tin mô hình sau khi áp dụng PEFT:")
    model.print_trainable_parameters()


    print("\n>>> Bước 2: Chuẩn bị Dữ liệu...")
    
    dataset = build_dataset(args.dataset, args.root_path, args.shots, processor.image_processor)
    
    train_loader = build_data_loader(
        data_source=dataset.train_x,
        batch_size=args.batch_size,
        tfm=processor.image_processor, # Dùng transform cơ bản cho train
        is_train=True,
        shuffle=True
    )
    eval_batch_size = args.batch_size * 2 
    val_loader = build_data_loader(
        data_source=dataset.val,
        batch_size=eval_batch_size,
        tfm=processor.image_processor,
        is_train=False,
        shuffle=False
    )
    test_loader = build_data_loader(
        data_source=dataset.test,
        batch_size=eval_batch_size,
        tfm=processor.image_processor,
        is_train=False,
        shuffle=False
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