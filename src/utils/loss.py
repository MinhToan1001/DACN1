import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss với hỗ trợ đầy đủ cho class-weight Tensor.

    [YÊU CẦU 2] Sửa lại tham số alpha:
    - alpha có thể là:
        * None          → không dùng class weighting
        * float         → nhân toàn bộ loss với 1 hằng số (ít dùng)
        * torch.Tensor  → mảng trọng số shape [num_classes], mỗi class 1 trọng số
    - Trong forward, alpha_t được gather đúng theo targets để:
        * Phạt nặng class đa số (600 ảnh → weight nhỏ)
        * Buff mạnh class thiểu số (3-9 ảnh → weight lớn)

    Công thức: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Tham khảo: Lin et al. (2017) - "Focal Loss for Dense Object Detection"
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: None | float | torch.Tensor shape [num_classes]
                   Nên truyền vào Tensor class_weights từ pipeline để tối ưu.
            gamma: Focusing parameter. gamma=2 là khuyến nghị từ paper gốc.
            reduction: 'mean' | 'sum' | 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  Tensor shape [B, num_classes] - raw logits (chưa softmax)
            targets: Tensor shape [B]              - ground-truth class indices

        Returns:
            Scalar loss (nếu reduction='mean'/'sum') hoặc Tensor [B] nếu 'none'
        """
        # ----------------------------------------------------------------
        # Bước 1: Tính Cross-Entropy tiêu chuẩn (per-sample, không reduction)
        # KHÔNG truyền weight vào F.cross_entropy ở đây vì ta sẽ áp dụng
        # alpha thủ công sau (để gather đúng alpha_t cho từng sample).
        # ----------------------------------------------------------------
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # ----------------------------------------------------------------
        # Bước 2: Tính p_t = exp(-CE) = xác suất model đoán đúng class thật
        # p_t cao → sample dễ → focal weight thấp → giảm đóng góp vào loss
        # p_t thấp → sample khó → focal weight cao → tập trung học hơn
        # ----------------------------------------------------------------
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # ----------------------------------------------------------------
        # Bước 3: Áp dụng alpha (class weighting)
        # ----------------------------------------------------------------
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Đảm bảo alpha Tensor nằm trên cùng device với targets
                alpha = self.alpha.to(inputs.device)
                # gather: lấy đúng trọng số của class thật cho mỗi sample
                # targets.view(-1) có shape [B], alpha_t có shape [B]
                alpha_t = alpha.gather(0, targets.view(-1))
                focal_loss = alpha_t * focal_loss
            else:
                # alpha là float scalar
                focal_loss = float(self.alpha) * focal_loss

        # ----------------------------------------------------------------
        # Bước 4: Reduction
        # ----------------------------------------------------------------
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss  # 'none': trả về Tensor [B]