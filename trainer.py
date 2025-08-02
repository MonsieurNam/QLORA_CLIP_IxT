# @title trainer.py – Sử dụng Adam8bit (không fused)

import time
from typing import Dict
from xml.parsers.expat import model

import torch
import torch.nn.functional as F
from tqdm import tqdm
from bitsandbytes.optim import Adam8bit # <-- Thay đổi optimizer

from metrics import cls_acc

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def format_time(sec: float) -> str:
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(self, args, model, processor, dataset,
                 train_loader, val_loader, test_loader):
        self.args = args
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        print(">>> Sử dụng Adam8bit Optimizer (không fused).")
        self.optimizer = Adam8bit(
            model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-5, weight_decay=1e-2,
        )

        self.total_updates = args.n_iters * args.shots
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.total_updates, eta_min=1e-6
        )

        if args.compute_dtype == "auto":
            self.compute_dtype = (
                torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            self.compute_dtype = torch.bfloat16 if args.compute_dtype == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.compute_dtype == torch.float16)

        template = dataset.template[0]
        prompts = [template.format(c.replace("_", " ")) for c in dataset.classnames]
        txt_inputs: Dict[str, torch.Tensor] = processor(
            text=prompts, return_tensors="pt", padding=True, truncation=True)
        txt_inputs = {k: v.to(model.device) for k, v in txt_inputs.items()}

        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.compute_dtype):
            text_feat = model.get_text_features(**txt_inputs)
        self.text_feat = (text_feat / text_feat.norm(dim=-1, keepdim=True)).to(self.compute_dtype).cuda()       
        model.train()

    def train(self):
        eff_bs = self.args.batch_size * self.args.gradient_accumulation_steps
        print("\n===== Bắt đầu Huấn luyện =====")
        print(f"Số update: {self.total_updates} | micro-bs={self.args.batch_size} | eff-bs={eff_bs}")

        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        start = time.time()

        upd, epoch = 0, 0
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        while upd < self.total_updates:
            epoch += 1
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for i, (imgs, tgt) in enumerate(pbar):
                if upd >= self.total_updates: break

                imgs, tgt = imgs.to(self.model.device), tgt.to(self.model.device)
                with torch.amp.autocast("cuda", dtype=self.compute_dtype):
                    img_feat = self.model.get_image_features(pixel_values=imgs)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    text_feat = self.text_feat
                    logits = (self.model.logit_scale.exp().to(img_feat.dtype) * img_feat @ text_feat.t())
                    loss = F.cross_entropy(logits, tgt) / self.args.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if ((i + 1) % self.args.gradient_accumulation_steps == 0 or i + 1 == len(self.train_loader)):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    upd += 1
                    pbar.set_postfix({"upd": f"{upd}/{self.total_updates}", "lr": f"{self.scheduler.get_last_lr()[0]:.1e}"})
                    if upd >= self.total_updates: break

        runtime, peak = format_time(time.time() - start), torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"\n--- Stats ---\nTime : {runtime}\nVRAM : {peak:.2f} GB (peak)")
        self.evaluate("test")

    def evaluate(self, split="test"):
        print(f"\nĐánh giá trên {split.upper()}")
        self.model.eval()
        loader = self.test_loader if split == "test" else self.val_loader

        total_acc = total = 0.0
        with torch.no_grad():
            for imgs, tgt in tqdm(loader):
                imgs, tgt = imgs.to(self.model.device), tgt.to(self.model.device)
                with torch.amp.autocast("cuda", dtype=self.compute_dtype):
                    img_feat = self.model.get_image_features(pixel_values=imgs)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    text_feat = self.text_feat.to(img_feat.dtype)
                    logits = (self.model.logit_scale.exp().to(img_feat.dtype) * img_feat @ text_feat.t())

                batch_acc = cls_acc(logits, tgt)
                if batch_acc > 1: batch_acc /= 100.0
                total_acc += batch_acc * tgt.size(0)
                total += tgt.size(0)

        acc = total_acc / total
        print(f"**** Kết quả: {acc * 100:.2f}% ****")
        self.model.train()
        return acc