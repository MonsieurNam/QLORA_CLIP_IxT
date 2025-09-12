# @title run_utils.py
import argparse
import random
import numpy as np
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    
def print_gpu_memory_usage(stage=""):
    """In ra mức sử dụng VRAM đỉnh điểm từ PyTorch."""
    if torch.cuda.is_available():
        peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        print(f"[{stage}] Peak PyTorch VRAM: {peak_allocated_gb:.2f} GB (Allocated) | {peak_reserved_gb:.2f} GB (Reserved)")

def set_random_seed(seed):
    """Thiết lập random seed để đảm bảo tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_system_vram_usage():
    """Lấy mức sử dụng VRAM từ driver NVIDIA (giống nvidia-smi)."""
    if not NVML_AVAILABLE or not torch.cuda.is_available():
        if not NVML_AVAILABLE:
            print("Cảnh báo: Thư viện 'pynvml' chưa được cài đặt. Không thể đo VRAM hệ thống. Hãy chạy 'pip install pynvml'.")
        return None, None
    
    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_gb = mem_info.used / (1024**3)
    total_gb = mem_info.total / (1024**3)
    return used_gb, total_gb

def get_arguments():
    """Định nghĩa và phân tích các tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="QLoRA/LoRA fine-tuning for CLIP models")

    # --- Cấu hình Thí nghiệm Cơ bản ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed cho thí nghiệm.')
    parser.add_argument('--dataset', type=str, required=True, help='Tên dataset để huấn luyện (ví dụ: dtd, ucf101).')
    parser.add_argument('--root_path', type=str, default='./data', help='Đường dẫn đến thư mục chứa dữ liệu.')
    parser.add_argument('--shots', type=int, default=16, help='Số lượng mẫu few-shot cho mỗi lớp.')
    # parser.add_argument('--backbone', type=str, default='ViT-B/16', choices=['ViT-B/16', 'ViT-B/32', 'ViT-L/14','ViT-g-14','siglip-so400m'], help='Kiến trúc CLIP backbone.')
    parser.add_argument('--backbone', type=str, default='ViT-B/16',
                        help='Tên viết tắt (ví dụ: ViT-g-14), repo_id, hoặc đường dẫn đến thư mục cục bộ.')
    # --- Cấu hình Chế độ Huấn luyện ---
    parser.add_argument('--mode', type=str, default='qlora', choices=['qlora', 'lora'],
                        help="Chọn chế độ: 'qlora' (lượng tử hóa 4-bit) hoặc 'lora' (float16/bfloat16).")

    # --- Cấu hình LoRA / DoRA ---
    parser.add_argument('--lora_target_modules', nargs='+', default=['q_proj', 'v_proj'],
                        help="Danh sách các module để áp dụng LoRA. Ví dụ: 'q_proj' 'k_proj' 'v_proj'. "
                             "Để có hiệu năng QLoRA tốt nhất, hãy thử: 'q_proj' 'k_proj' 'v_proj' 'out_proj' 'fc1' 'fc2'.")
    parser.add_argument('--r', type=int, default=8, help='Rank của ma trận LoRA.')
    parser.add_argument('--alpha', type=int, default=16, help='Hệ số scale alpha của LoRA.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Tỷ lệ dropout của LoRA.')
    parser.add_argument('--use_dora', action='store_true', help='Sử dụng DoRA thay cho LoRA.')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                    help='Bật Gradient Checkpointing để tiết kiệm VRAM.')

    # --- Cấu hình Quá trình Huấn luyện ---
    parser.add_argument('--n_iters', type=int, default=10000, help='Tổng số vòng lặp huấn luyện.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size cho training loader.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Số bước tích lũy gradient.')
    parser.add_argument('--compute_dtype', type=str, default='auto', choices=['auto', 'fp16', 'bf16'], help='Compute dtype (float16 hoặc bfloat16).')

    # --- Cấu hình Lưu trữ & Đánh giá ---
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Đường dẫn để lưu adapter đã huấn luyện.')
    parser.add_argument('--eval_only', action='store_true', help='Chỉ chạy đánh giá trên mô hình đã lưu.')

    args = parser.parse_args()

    # In thông báo xác nhận chế độ đang chạy
    print("--- Phân tích Cấu hình ---")
    if args.mode == 'lora':
        print("-> Chế độ: Classic LoRA (giống paper CLIP-LORA)")
        print("-> Mô hình base sẽ được tải ở float16/bfloat16.")
    else: # qlora
        print("-> Chế độ: QLoRA (giống paper QLoRA)")
        print("-> Mô hình base sẽ được lượng tử hóa 4-bit.")
        all_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
        # Sắp xếp để so sánh không phụ thuộc thứ tự
        if sorted(args.lora_target_modules) != sorted(all_modules):
            print(f"Cảnh báo: Chế độ QLoRA thường cho kết quả tốt nhất khi target tất cả các lớp linear.")
            print(f"   Bạn đang target: {args.lora_target_modules}")
            print(f"   Để có hiệu suất tối ưu, hãy xem xét dùng: --lora_target_modules {' '.join(all_modules)}")
    print("-------------------------")

    return args