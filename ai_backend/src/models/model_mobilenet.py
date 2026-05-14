import torch
import torch.nn as nn
from torchvision import models


def build_mobilenetv2_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Xây dựng MobileNetV2 với chiến lược fine-tuning có chọn lọc.

    [YÊU CẦU 2] Freeze/Unfreeze:
    - MobileNetV2 có 19 InvertedResidual blocks (features[0] đến features[18])
      + features[0] là Conv2d đầu vào.
    - FREEZE: features[0] đến features[14] (bao gồm) — các lớp đặc trưng thấp & trung
    - UNFREEZE: features[15] đến features[18] (4 blocks cuối) — đặc trưng cao cấp
    - UNFREEZE: toàn bộ classifier mới

    Tại sao chọn unfreeze từ block 15?
    - Block 15-18 học các đặc trưng ngữ nghĩa cao (texture, shape phức tạp) —
      cần fine-tune để thích nghi với 103 loài động vật quý hiếm.
    - Block 0-14 học edge, color, texture cơ bản — đã tốt với ImageNet weights,
      không cần cập nhật, tiết kiệm memory + tránh catastrophic forgetting.

    [YÊU CẦU 2] Classifier:
    - Dropout(p=0.5): tăng so với p=0.2 gốc để chống overfit mạnh hơn
      (dataset nhỏ ~15k ảnh, 103 classes → nguy cơ overfit cao)
    - Linear(in_features, num_classes): lớp phân loại cuối

    Args:
        num_classes: Số lớp cần phân loại (103 loài)
        pretrained:  Có tải pretrained ImageNet weights không

    Returns:
        model: MobileNetV2 đã cấu hình sẵn
    """
    # ----------------------------------------------------------------
    # Bước 1: Tải pretrained MobileNetV2
    # ----------------------------------------------------------------
    if pretrained:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    else:
        model = models.mobilenet_v2()

    # ----------------------------------------------------------------
    # [YÊU CẦU 2] Bước 2: Freeze TẤT CẢ tham số trước
    # Sau đó chỉ unfreeze phần cần fine-tune (selective unfreezing)
    # ----------------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False

    # ----------------------------------------------------------------
    # [YÊU CẦU 2] Bước 3: Unfreeze features[15] đến features[18]
    # MobileNetV2 features có 19 phần tử (index 0-18):
    #   features[0]     : Conv2d ban đầu (3→32)
    #   features[1-18]  : 18 InvertedResidual blocks
    # Unfreeze 4 blocks cuối (index 15, 16, 17, 18)
    # ----------------------------------------------------------------
    UNFREEZE_FROM = 10  # Unfreeze từ block 15 trở đi (inclusive)
    for i in range(UNFREEZE_FROM, len(model.features)):
        for param in model.features[i].parameters():
            param.requires_grad = True

    # ----------------------------------------------------------------
    # [YÊU CẦU 2] Bước 4: Thay classifier với Dropout(0.5) + Linear mới
    # Classifier mới tự động có requires_grad=True
    # ----------------------------------------------------------------
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),               # Tăng từ 0.2 → 0.5 để chống overfit
        nn.Linear(in_features, num_classes)
    )

    # ----------------------------------------------------------------
    # Kiểm tra số params được train
    # ----------------------------------------------------------------
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params   = total_params - trainable_params

    print(f"✅ MobileNetV2 đã cấu hình:")
    print(f"   Freeze: features[0] → features[{UNFREEZE_FROM - 1}]")
    print(f"   Unfreeze: features[{UNFREEZE_FROM}] → features[{len(model.features) - 1}] + classifier")
    print(f"   Tổng params:     {total_params:>12,}")
    print(f"   Frozen params:   {frozen_params:>12,}  (không train)")
    print(f"   Trainable params:{trainable_params:>12,}  (được train)")

    return model