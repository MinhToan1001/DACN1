import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss với hỗ trợ đầy đủ cho:
    - Nhãn đơn (hard labels): Tensor [B] kiểu long — dùng cho Val/Test
    - Nhãn MixUp (soft/mixed targets): được xử lý bên ngoài ở train_epoch
      bằng công thức: loss = lam * criterion(outputs, targets_a) + (1-lam) * criterion(outputs, targets_b)
      Cả targets_a và targets_b đều là hard labels [B] → FocalLoss nhận được
      hard labels ở mọi lúc, không cần thay đổi internal logic.

    [YÊU CẦU 4] Tham số alpha:
    - None:           Không dùng class weighting
    - float:          Nhân toàn bộ loss với 1 hằng số
    - torch.Tensor:   Mảng trọng số shape [num_classes], mỗi class 1 trọng số
                      → gather đúng alpha_t theo targets cho từng sample

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
            targets: Tensor shape [B] - ground-truth class indices (hard labels)
                     Khi dùng với MixUp: gọi 2 lần riêng biệt với targets_a và targets_b,
                     rồi kết hợp bên ngoài: loss = lam * FL(out, a) + (1-lam) * FL(out, b)

        Returns:
            Scalar loss (nếu reduction='mean'/'sum') hoặc Tensor [B] nếu 'none'
        """
        # ----------------------------------------------------------------
        # Bước 1: Tính Cross-Entropy per-sample (không reduction)
        # KHÔNG truyền weight vào F.cross_entropy để áp dụng alpha thủ công
        # sau (gather đúng alpha_t cho từng sample theo targets).
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
                # Đảm bảo alpha Tensor nằm trên cùng device với inputs
                alpha = self.alpha.to(inputs.device)
                # gather: lấy đúng trọng số của class thật cho mỗi sample
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