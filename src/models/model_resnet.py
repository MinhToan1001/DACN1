import torch.nn as nn
from torchvision import models

def build_resnet50_model(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    
    # Lấy số lượng features của lớp cuối cùng
    num_ftrs = model.fc.in_features
    
    # Thay thế bằng lớp mới tương ứng với số class động vật của bạn
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model