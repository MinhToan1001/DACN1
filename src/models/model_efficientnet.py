import torch.nn as nn
from torchvision import models

def build_efficientnet_b0_model(num_classes, pretrained=True):
    model = models.efficientnet_b0(pretrained=pretrained)
    
    # Lấy số lượng features của lớp cuối cùng
    num_ftrs = model.classifier[1].in_features
    
    # Thay thế bằng lớp mới tương ứng với số class động vật của bạn
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model