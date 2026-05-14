import torch.nn as nn
from torchvision import models

def build_efficientnet_b3_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Xây dựng EfficientNet-B3 với cơ chế Squeeze-and-Excitation (SE) gốc. 
    Bù đắp điểm yếu của ResNet50 ở việc tập trung cục bộ (Local Attention).

    Chiến lược Gradual Unfreezing:
    ┌─────────────────────────────────────────────────────────┐
    │  Layer               │ Trạng thái      │ Lý do               │
    ├─────────────────────────────────────────────────────────┤
    │  features[0]-[5]     │ FROZEN          │ Học low & mid-level │
    │  features[6]-[8]     │ UNFROZEN ✓      │ High-level features │
    │  classifier (mới)    │ UNFROZEN ✓      │ Phân loại classes   │
    └─────────────────────────────────────────────────────────┘
    """
    model = models.efficientnet_b3(pretrained=pretrained)

    # 1. Freeze TOÀN BỘ model trước
    for param in model.parameters():
        param.requires_grad = False

    # 2. Unfreeze block cuối (từ features[6] đến hết)
    # Đây là block học đặc trưng chi tiết nhỏ, cực kỳ cần thiết cho động vật 
    for i in range(6, len(model.features)):
        for param in model.features[i].parameters():
            param.requires_grad = True

    # 3. Thay thế Classifier layer 
    num_ftrs = model.classifier[1].in_features  # 1536 cho B3
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"[EfficientNet-B3] Tổng params:      {total_params:,}")
    print(f"[EfficientNet-B3] Trainable params: {trainable_params:,}  (features[6:] + classifier)")
    print(f"[EfficientNet-B3] Frozen params:    {frozen_params:,}  (features[0~5])")

    return model