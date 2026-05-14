import torch.nn as nn
from torchvision import models


def build_resnet50_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Xây dựng ResNet50 với Gradual Unfreezing và Dropout để chống Overfitting.

    [YÊU CẦU 3] Chiến lược Gradual Unfreezing:
    ┌─────────────────────────────────────────────────────────┐
    │  Layer          │ Trạng thái      │ Lý do              │
    ├─────────────────────────────────────────────────────────┤
    │  conv1, bn1     │ FROZEN          │ Học low-level edges │
    │  layer1         │ FROZEN          │ Học textures cơ bản │
    │  layer2         │ FROZEN          │ Học mid-level feats │
    │  layer3         │ FROZEN          │ Học high-level feats│
    │  layer4         │ UNFROZEN ✓      │ Fine-tune domain    │
    │  fc (mới)       │ UNFROZEN ✓      │ Phân loại classes   │
    └─────────────────────────────────────────────────────────┘

    Lý do freeze layer1-3:
    - Các layer này đã học được đặc trưng tổng quát từ ImageNet
    - Freeze giúp tránh catastrophic forgetting và giảm overfit
    - Chỉ cần fine-tune layer4 (semantic features) cho domain mới

    Dropout(p=0.5) trong FC:
    - Ngăn model memorize training samples (đặc biệt class hiếm 3-9 ảnh)
    - Forcing redundant representations → generalization tốt hơn

    Args:
        num_classes: Số class động vật cần phân loại
        pretrained:  True để dùng ImageNet weights (luôn nên True khi ít data)

    Returns:
        model với ~10M params được train (layer4 + FC), ~13M params frozen
    """
    # Load pretrained ResNet50
    model = models.resnet50(pretrained=pretrained)

    # ----------------------------------------------------------------
    # Bước 1: Freeze TOÀN BỘ model trước
    # ----------------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False

    # ----------------------------------------------------------------
    # Bước 2: Unfreeze layer4 (block cuối của ResNet, học semantic features)
    # Đây là layer học các đặc trưng phức tạp nhất → cần fine-tune cho
    # domain động vật quý hiếm (khác xa ImageNet)
    # ----------------------------------------------------------------
    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    # ----------------------------------------------------------------
    # Bước 3: Thay thế FC layer với Dropout + Linear mới
    # - Dropout(0.5): trong mỗi forward pass, 50% neurons bị tắt ngẫu nhiên
    #   → model không thể dựa vào bất kỳ neuron nào quá nhiều → chống memorize
    # - Linear(num_ftrs, num_classes): output đúng số class của bài toán
    # ----------------------------------------------------------------
    num_ftrs = model.fc.in_features  # ResNet50: 2048
    model.fc = nn.Sequential(
            nn.Dropout(p=0.3),         
            nn.Linear(num_ftrs, 512),    
            nn.ReLU(),                   
            nn.Dropout(p=0.2),          
            nn.Linear(512, num_classes) 
        )
  
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params   = total_params - trainable_params

    print(f"[ResNet50] Tổng params:      {total_params:,}")
    print(f"[ResNet50] Trainable params: {trainable_params:,}  (layer4 + FC)")
    print(f"[ResNet50] Frozen params:    {frozen_params:,}  (conv1~layer3)")

    return model