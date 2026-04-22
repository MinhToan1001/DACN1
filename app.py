import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.model_mobilenet import build_mobilenetv2_model
from src.models.model_efficientnet import build_efficientnet_b3_model
from src.models.model_resnet import build_resnet50_model
from src.utils.loss import FocalLoss
from src.utils.train import ModelTrainer
from src.data.preprocess import RareAnimalPipeline
import matplotlib
matplotlib.use('Agg')


def main():
    # ================================================================
    # [CẤU HÌNH KIẾN TRÚC]
    # Hãy đổi giá trị này thành tên model bạn muốn train luân phiên:
    # Lựa chọn: "resnet50", "efficientnet_b3", "mobilenetv2"
    # ================================================================
    MODEL_TO_TRAIN = "efficientnet_b3"  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 20

    # ----------------------------------------------------------------
    # Bước 1: Chạy pipeline → lấy DataLoaders + class_weights
    # ----------------------------------------------------------------
    print("⏳ Đang chuẩn bị dữ liệu qua Pipeline...")
    pipeline = RareAnimalPipeline(
        data_dir="images",
        image_size=224,
        batch_size=32,
        use_focal_loss=True
    )
    result = pipeline.run()

    train_loader = result['train_loader']
    val_loader   = result['val_loader']
    test_loader  = result['test_loader']
    class_names  = result['class_names']
    NUM_CLASSES  = result['num_classes']

    # ----------------------------------------------------------------
    # Nhận class_weights từ pipeline và chuyển lên device
    # class_weights là Tensor [num_classes] tính bằng effective_num:
    #   - class nhiều ảnh → weight nhỏ (phạt nhẹ)
    #   - class ít ảnh   → weight lớn (buff mạnh)
    # ----------------------------------------------------------------
    class_weights = result['class_weights'].to(device)
    print(f"✅ Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")

    # ----------------------------------------------------------------
    # Truyền class_weights Tensor vào FocalLoss
    # Kết hợp focal mechanism + class weighting = double protection
    # ----------------------------------------------------------------
    criterion = FocalLoss(alpha=class_weights, gamma=2)  # [YÊU CẦU 4] gamma=0.5 để giảm bớt hiệu ứng triệt tiêu của focal loss

    # ----------------------------------------------------------------
    # Bước 2: Khởi tạo model và Optimizer động theo lựa chọn
    # ----------------------------------------------------------------
    if MODEL_TO_TRAIN == "efficientnet_b3":
        print(f"🧠 Đang khởi tạo EfficientNet-B3 cho {NUM_CLASSES} classes...")
        model = build_efficientnet_b3_model(num_classes=NUM_CLASSES, pretrained=True)
        model = model.to(device)
        
        optimizer = torch.optim.AdamW([
            {'params': model.features[6].parameters(), 'lr': 1e-5},
            {'params': model.features[7].parameters(), 'lr': 3e-5},
            {'params': model.features[8].parameters(), 'lr': 5e-5},
            {'params': model.classifier.parameters(),  'lr': 1e-4}
        ], weight_decay=0.01)

    elif MODEL_TO_TRAIN == "resnet50":
        print(f"🧠 Đang khởi tạo ResNet50 cho {NUM_CLASSES} classes...")
        model = build_resnet50_model(num_classes=NUM_CLASSES, pretrained=True)
        model = model.to(device)

        optimizer = torch.optim.AdamW([
            {'params': model.layer3.parameters(), 'lr': 1e-5},
            {'params': model.layer4.parameters(), 'lr': 5e-5},
            {'params': model.fc.parameters(),     'lr': 1e-4}
        ], weight_decay=0.01)
        
    elif MODEL_TO_TRAIN == "mobilenetv2":
        print(f"🧠 Đang khởi tạo MobileNetV2 cho {NUM_CLASSES} classes...")
        model = build_mobilenetv2_model(num_classes=NUM_CLASSES, pretrained=True)
        model = model.to(device)
        
        # Chỉ định unfreeze phần backend của MobileNet
        optimizer = torch.optim.AdamW([
            {'params': model.features[15:].parameters(), 'lr': 5e-5}, 
            {'params': model.classifier.parameters(),   'lr': 1e-4}
        ], weight_decay=0.01)
    else:
        raise ValueError(f"Không hỗ trợ mô hình '{MODEL_TO_TRAIN}'. Hãy chọn resnet50, efficientnet_b3, hoặc mobilenetv2.")

    # ----------------------------------------------------------------
    # Chỉ truyền params có requires_grad=True (Chỉ để in log, do optimizer đã nạp ở trên)
    # ----------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"🔧 Số param được optimize: {sum(p.numel() for p in trainable_params):,}")

    # ----------------------------------------------------------------
    # ReduceLROnPlateau Scheduler:
    # - mode='min': theo dõi val_loss, giảm LR khi val_loss không cải thiện
    # - factor=0.5: LR mới = LR cũ * 0.5
    # - patience=2: chờ 2 epoch không cải thiện mới giảm LR
    # ----------------------------------------------------------------
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )

    # ----------------------------------------------------------------
    # Bước 3: Training
    # [YÊU CẦU 5] ModelTrainer.fit() sẽ:
    #   - Vòng lặp epoch: chỉ chạy train + val
    #   - Sau khi xong: load best model → test 1 lần duy nhất
    # ----------------------------------------------------------------
    print("🚀 Bắt đầu huấn luyện...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        class_names=class_names
    )

    trainer.fit(epochs=EPOCHS)


if __name__ == "__main__":
    main()