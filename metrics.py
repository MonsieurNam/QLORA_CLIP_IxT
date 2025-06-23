import torch

def cls_acc(output: torch.Tensor, target: torch.Tensor, topk: int = 1) -> float:
    """
    Tính toán độ chính xác top-k cho một bài toán phân loại.

    Hàm này nhận đầu ra logits của mô hình và nhãn đúng, sau đó
    tính toán tỷ lệ phần trăm các mẫu được dự đoán chính xác.

    Args:
        output (torch.Tensor): Logits đầu ra từ mô hình.
                               Đây là một tensor 2D với shape (batch_size, num_classes).
        target (torch.Tensor): Nhãn đúng (ground truth) cho mỗi mẫu.
                               Đây là một tensor 1D với shape (batch_size).
        topk (int, optional): Giá trị k để tính top-k accuracy. 
                              Mặc định là 1 (tức là Top-1 accuracy).

    Returns:
        float: Độ chính xác được tính bằng phần trăm (từ 0 đến 100).
    """
    with torch.no_grad():
        pred = output.topk(topk, 1, True, True)[1]
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
        res = (correct_k.mul_(100.0 / target.shape[0]))
        
    return res.item()