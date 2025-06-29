# %%writefile /content/QLORA_CLIP_IxT/trainer.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import time

from metrics import cls_acc

def format_time(seconds):
    """Chuyển đổi giây thành định dạng Giờ:Phút:Giây."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

class Trainer:
    """
    Lớp đóng gói logic huấn luyện và đánh giá cho mô hình CLIP với QLoRA.
    """
    def __init__(self, args, model, processor, dataset, train_loader, val_loader, test_loader):
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
            weight_decay=0.01
        )

        self.total_iters = args.n_iters * args.shots

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_iters,
            eta_min=1e-6
        )

        self._cached_text_features = None

    def _get_text_features(self):
        if self._cached_text_features is None:
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
        print("\n===== Bắt đầu Huấn luyện =====")
        print(f"Tổng số vòng lặp (iterations): {self.total_iters}")
        print(f"Effective batch size: {self.args.batch_size * self.args.gradient_accumulation_steps}")

        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        count_iters = 0
        self.model.train()

        num_epochs = (self.total_iters // len(self.train_loader)) + 1

        for epoch in range(num_epochs):
            if count_iters >= self.total_iters:
                break

            epoch_loss = 0.
            epoch_acc = 0.
            epoch_samples = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} | Iter {count_iters}")
            for i, batch in enumerate(progress_bar):
                if count_iters >= self.total_iters:
                    break

                images, target = batch
                images = images.to(self.model.device)
                target = target.to(self.model.device)

                image_features = self.model.get_image_features(pixel_values=images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                text_features = self._get_text_features()

                logit_scale = self.model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()

                loss = F.cross_entropy(logits, target)

                current_loss = loss.item()
                epoch_loss += current_loss * len(target)
                epoch_acc += cls_acc(logits, target) * len(target)
                epoch_samples += len(target)

                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                progress_bar.set_postfix({
                    "Loss": f"{current_loss:.4f}",
                    "LR": f"{self.scheduler.get_last_lr()[0]:.1e}"
                })

                if (i + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                count_iters += 1

            avg_epoch_loss = epoch_loss / epoch_samples
            avg_epoch_acc = epoch_acc / epoch_samples
            current_lr = self.scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1} done. LR: {current_lr:.6f}, Train Acc: {avg_epoch_acc:.2f}%, Avg Loss: {avg_epoch_loss:.4f}')

        end_time = time.time()
        total_training_time = end_time - start_time

        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            peak_vram_gb = 0

        print("\n--- Thống kê Tài nguyên và Kết quả ---")
        print(f"Tổng thời gian huấn luyện: {format_time(total_training_time)}")
        print(f"Peak VRAM sử dụng: {peak_vram_gb:.2f} GB")

        final_test_acc = self.evaluate("test")
        print("------------------------------------")


        if self.args.save_path:
            save_dir = os.path.join(self.args.save_path, f"{self.args.dataset}_{self.args.shots}shots")
            print(f"Lưu adapter LoRA đã huấn luyện vào '{save_dir}'")
            self.model.save_pretrained(save_dir)

    def evaluate(self, split="test"):
        print(f"\n===== Bắt đầu Đánh giá trên tập {split.upper()} =====")
        self.model.eval()
        loader = self.test_loader if split == "test" else self.val_loader
        total_acc, total_samples = 0., 0

        with torch.no_grad():
            text_features = self._get_text_features()
            for batch in tqdm(loader, desc=f"Đang đánh giá trên tập {split}"):
                images, target = batch
                images = images.to(self.model.device)
                target = target.to(self.model.device)

                image_features = self.model.get_image_features(pixel_values=images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                logit_scale = self.model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()

                acc = cls_acc(logits, target)
                total_acc += acc * len(target)
                total_samples += len(target)

        final_acc = total_acc / total_samples
        print(f"**** Kết quả: Độ chính xác trên tập {split.upper()} = {final_acc:.2f}% ****")
        return final_acc