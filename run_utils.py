import argparse
import random
import numpy as np
import torch

def set_random_seed(seed):
    """
    Thiết lập seed cho các thư viện sinh số ngẫu nhiên để đảm bảo
    kết quả có thể được tái tạo.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_arguments():
    """
    Phân tích và trả về các tham số được truyền vào từ dòng lệnh.
    """
    parser = argparse.ArgumentParser(description="Tinh chỉnh Few-shot cho CLIP bằng QLoRA")

    group_dataset = parser.add_argument_group('Cấu hình Dataset')
    group_dataset.add_argument('--root_path', type=str, default='./data', 
                               help='Đường dẫn đến thư mục gốc chứa tất cả các bộ dữ liệu.')
    group_dataset.add_argument('--dataset', type=str, default='dtd',
                               choices=['dtd', 'eurosat', 'caltech101', 'food101', 'oxford_pets', 
                                        'stanford_cars', 'oxford_flowers', 'sun397', 'ucf101', 
                                        'imagenet', 'fgvc'],
                               help='Tên bộ dữ liệu để sử dụng.')
    group_dataset.add_argument('--shots', default=16, type=int, 
                               help='Số lượng mẫu cho mỗi lớp trong few-shot learning (K-shot).')

    group_model = parser.add_argument_group('Cấu hình Mô hình và PEFT')
    group_model.add_argument('--backbone', default='ViT-B/16', type=str, 
                             choices=['ViT-B/16', 'ViT-B/32', 'ViT-L/14'],
                             help='Kiến trúc backbone của CLIP để sử dụng.')
    group_model.add_argument('--r', default=2, type=int, 
                             help='Rank (thứ hạng) của các ma trận LoRA.')
    group_model.add_argument('--alpha', default=1, type=int, 
                             help='Tham số alpha để co giãn trong LoRA (lora_alpha).')
    group_model.add_argument('--dropout_rate', default=0.25, type=float, 
                             help='Tỷ lệ dropout của LoRA (lora_dropout).')
    group_model.add_argument('--params', metavar='P', type=str, nargs='+', default=['q', 'v'], 
                             help="Các ma trận trong khối attention để áp dụng LoRA. Ví dụ: 'q' 'v' 'k' 'o'.")

    group_train = parser.add_argument_group('Cấu hình Huấn luyện')
    group_train.add_argument('--lr', default=2e-4, type=float, 
                             help='Learning rate ban đầu.')
    group_train.add_argument('--n_iters', default=500, type=int, 
                             help='Hệ số nhân cho tổng số vòng lặp (total_iters = n_iters * shots).')
    group_train.add_argument('--batch_size', default=32, type=int, 
                             help='Kích thước batch cho mỗi lần cập nhật GPU.')
    group_train.add_argument('--gradient_accumulation_steps', default=1, type=int,
                             help='Số bước tích lũy gradient để mô phỏng batch size lớn hơn.')

    group_system = parser.add_argument_group('Cấu hình Hệ thống và Chế độ chạy')
    group_system.add_argument('--seed', default=1, type=int, 
                              help='Seed ngẫu nhiên để tái tạo kết quả.')
    group_system.add_argument('--compute_dtype', type=str, default='auto', choices=['auto', 'fp16', 'bf16'],
                              help="Kiểu dữ liệu tính toán cho QLoRA ('auto' sẽ tự động chọn).")
    group_system.add_argument('--save_path', default=None, type=str,
                              help='Đường dẫn để lưu adapter LoRA đã huấn luyện. Không lưu nếu để trống.')
    group_system.add_argument('--eval_only', default=False, action='store_true', 
                              help='Cờ để chỉ chạy đánh giá, không huấn luyện.')
    
    args = parser.parse_args()
    
    return args