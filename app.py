import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.model_resnet import build_resnet50_model
from src.utils.loss import FocalLoss
from src.utils.train import ModelTrainer
from src.data.preprocess import RareAnimalPipeline


def main():
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

    train_loader  = result['train_loader']
    val_loader    = result['val_loader']
    test_loader   = result['test_loader']
    class_names   = result['class_names']
    NUM_CLASSES   = result['num_classes']

    # ----------------------------------------------------------------
    # [YÊU CẦU 4a] Nhận class_weights từ pipeline và chuyển lên device
    # class_weights là Tensor [num_classes] tính bằng effective_num:
    #   - class 600 ảnh → weight ~0.1 (phạt nhẹ)
    #   - class 3 ảnh   → weight ~10+ (buff mạnh)
    # ----------------------------------------------------------------
    class_weights = result['class_weights'].to(device)
    print(f"✅ Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")

    # ----------------------------------------------------------------
    # [YÊU CẦU 4a] Truyền class_weights Tensor vào FocalLoss
    # Kết hợp focal mechanism + class weighting = double protection
    # cho class cực hiếm (3-9 ảnh)
    # ----------------------------------------------------------------
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # ----------------------------------------------------------------
    # Bước 2: Khởi tạo model (layer4 + FC đã unfreeze, layer1-3 frozen)
    # ----------------------------------------------------------------
    print(f"🧠 Đang khởi tạo ResNet50 cho {NUM_CLASSES} classes...")
    model = build_resnet50_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # ----------------------------------------------------------------
    # [YÊU CẦU 4b] Chỉ truyền params có requires_grad=True vào Optimizer
    # Lý do: Tránh lãng phí memory/compute tính gradient cho frozen layers
    # Lọc: layer4 (~8.4M params) + fc (~2M params) = ~10M params được train
    # ----------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"🔧 Số param được optimize: {sum(p.numel() for p in trainable_params):,}")

    # ----------------------------------------------------------------
    # [YÊU CẦU 4b] AdamW thay Adam:
    # - weight_decay=1e-2: L2 regularization decoupled từ gradient update
    #   (Adam chuẩn implement weight decay sai → AdamW fix điều này)
    # - Giúp penalize trọng số lớn → chống overfit thêm một lớp nữa
    # ----------------------------------------------------------------
    optimizer = optim.AdamW(trainable_params, lr=5e-5, weight_decay=1e-3)

    # ----------------------------------------------------------------
    # [YÊU CẦU 4c] ReduceLROnPlateau Scheduler:
    # - mode='min': theo dõi val_loss, giảm LR khi val_loss không cải thiện
    # - factor=0.5: LR mới = LR cũ * 0.5 (giảm một nửa)
    # - patience=3: chờ 3 epoch không cải thiện mới giảm LR
    # Tác dụng: Khi model bắt đầu overfit (val_loss tăng/đi ngang),
    #           LR giảm để model "bước nhỏ hơn" → thoát local minima
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