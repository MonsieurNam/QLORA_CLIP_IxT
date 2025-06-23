import torch
import torch.nn.functional as F
import time
import tqdm
from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers


def format_time(seconds):
    """Chuyển đổi giây thành định dạng Giờ:Phút:Giây."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    VALIDATION = False
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
 
    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    print("\n===== Bắt đầu Huấn luyện LoRA (Gốc) =====")
    print(f"Tổng số vòng lặp (iterations): {total_iters}")
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Sử dụng torch.amp.GradScaler theo cách mới để tránh warning
    scaler = torch.amp.GradScaler("cuda")
    count_iters = 0
    
    # --- LOGIC HUẤN LUYỆN ĐÃ ĐƯỢC SỬA ĐỔI HOÀN TOÀN ---
    optimizer.zero_grad() # Reset gradient một lần trước khi bắt đầu
    
    while count_iters < total_iters:
        # Vòng lặp này sẽ chạy qua train_loader nhiều lần nếu cần
        progress_bar = tqdm(train_loader, desc=f"Iter {count_iters}/{total_iters}")
        for i, (images, target) in enumerate(progress_bar):
            if count_iters >= total_iters:
                break

            clip_model.train() # Đặt ở đây để chắc chắn
            images, target = images.cuda(), target.cuda()
            
            # --- Forward pass ---
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Logic text encoder từ mã gốc
                if args.encoder == 'vision':
                    current_text_features = textual_features
                else:
                    texts = [dataset.template[0].format(c.replace('_', ' ')) for c in dataset.classnames]
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                    current_text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                
                cosine_similarity = logit_scale * image_features @ current_text_features.t()
                loss = F.cross_entropy(cosine_similarity, target)

            # --- Backward pass với Gradient Accumulation ---
            # Chuẩn hóa loss
            loss = loss / args.gradient_accumulation_steps
            
            # Tích lũy gradient
            scaler.scale(loss).backward()
            
            # Cập nhật scheduler mỗi step nhỏ
            scheduler.step()
            
            # Cập nhật optimizer chỉ sau một số bước nhất định
            if (count_iters + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            count_iters += 1
            progress_bar.set_description(f"Iter {count_iters}/{total_iters}")
            progress_bar.set_postfix({"Loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"})

    # --- Kết thúc đo lường và Logging ---
    end_time = time.time()
    total_training_time = end_time - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0

    print("\n--- Thống kê Tài nguyên (LoRA Gốc) ---")
    print(f"Tổng thời gian huấn luyện: {format_time(total_training_time)}")
    print(f"Peak VRAM sử dụng: {peak_vram_gb:.2f} GB")
    print("--------------------------------------")
    
    # Đánh giá cuối cùng
    acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
    print("**** Final test accuracy: {:.2f}%. ****\n".format(acc_test * 100))
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
        
    return