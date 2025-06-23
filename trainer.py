import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from metrics import cls_acc

class Trainer:
    """
    Lớp đóng gói logic huấn luyện và đánh giá cho mô hình CLIP với QLoRA.
    """
    def __init__(self, args, model, processor, dataset, train_loader, val_loader, test_loader):
        """
        Hàm khởi tạo.

        Args:
            args: Các tham số từ dòng lệnh.
            model: Mô hình PEFT (đã áp dụng QLoRA và LoRA).
            processor: Bộ xử lý từ Hugging Face (gồm image_processor và tokenizer).
            dataset: Đối tượng dataset từ module `datasets`.
            train_loader, val_loader, test_loader: Các DataLoader tương ứng.
        """
        self.args = args
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            betas=(0.9, 0.999), 
            weight_decay=0.01  # Sử dụng weight decay nhỏ
        )
        
        self.total_iters = args.n_iters * args.shots
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.total_iters, 
            eta_min=1e-6 # Learning rate tối thiểu
        )

        self._cached_text_features = None

    def _get_text_features(self):
        """Mã hóa và cache các đặc trưng text để không phải tính lại."""
        if self._cached_text_features is None:
            print("Đang tạo và cache các đặc trưng text...")
            texts = [self.dataset.template[0].format(c.replace('_', ' ')) for c in self.dataset.classnames]
            text_inputs = self.processor(
                text=texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.model.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
                self._cached_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return self._cached_text_features

    def train(self):
        """
        Thực hiện toàn bộ quá trình huấn luyện.
        """
        print("\n===== Bắt đầu Huấn luyện =====")
        print(f"Tổng số vòng lặp (iterations): {self.total_iters}")
        print(f"Effective batch size: {self.args.batch_size * self.args.gradient_accumulation_steps}")

        count_iters = 0
        self.model.train() # Chuyển mô hình sang chế độ huấn luyện
        
        while count_iters < self.total_iters:
            progress_bar = tqdm(self.train_loader, desc=f"Iter {count_iters}/{self.total_iters}")
            for i, batch in enumerate(progress_bar):
                if count_iters >= self.total_iters:
                    break
                images, target = batch
                target = target.to(self.model.device)

                image_inputs = self.processor(
                    images=images, 
                    return_tensors="pt"
                ).to(self.model.device)
                
                image_features = self.model.get_image_features(pixel_values=image_inputs["pixel_values"])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                text_features = self._get_text_features()
                
                logit_scale = self.model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()
                
                loss = F.cross_entropy(logits, target)
                
                loss = loss / self.args.gradient_accumulation_steps
                
                loss.backward()

                progress_bar.set_postfix({
                    "Loss": f"{loss.item() * self.args.gradient_accumulation_steps:.4f}",
                    "LR": f"{self.scheduler.get_last_lr()[0]:.1e}"
                })

                if (i + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                count_iters += 1
        
        print("\nHuấn luyện hoàn tất.")
        
        self.evaluate("test")
        
        if self.args.save_path:
            save_dir = os.path.join(self.args.save_path, f"{self.args.dataset}_{self.args.shots}shots")
            print(f"Lưu adapter LoRA đã huấn luyện vào '{save_dir}'")
            self.model.save_pretrained(save_dir)

    def evaluate(self, split="test"):
        """
        Đánh giá mô hình trên một tập dữ liệu cụ thể (val hoặc test).
        """
        print(f"\n===== Bắt đầu Đánh giá trên tập {split.upper()} =====")
        self.model.eval()  # Chuyển mô hình sang chế độ đánh giá
        
        loader = self.test_loader if split == "test" else self.val_loader
        
        total_acc = 0.
        total_samples = 0
        
        with torch.no_grad():
            text_features = self._get_text_features()

            for batch in tqdm(loader, desc=f"Đang đánh giá trên tập {split}"):
                images, target = batch
                target = target.to(self.model.device)
                
                image_inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
                image_features = self.model.get_image_features(pixel_values=image_inputs["pixel_values"])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                logit_scale = self.model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()
                
                acc = cls_acc(logits, target)
                total_acc += acc * len(target)
                total_samples += len(target)
        
        final_acc = total_acc / total_samples
        print(f"**** Kết quả: Độ chính xác trên tập {split.upper()} = {final_acc:.2f}% ****")
        return final_acc