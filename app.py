import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.model_mobilenet import build_mobilenetv2_model
from src.models.model_resnet import build_resnet50_model
from src.utils.loss import FocalLoss
from src.utils.train import ModelTrainer
from src.data.preprocess import RareAnimalPipeline
import matplotlib
matplotlib.use('Agg')


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
    # Bước 2: Khởi tạo model
    # [YÊU CẦU 2] features[0-14] frozen, features[15-18] + classifier unfreeze
    # ----------------------------------------------------------------
    print(f"🧠 Đang khởi tạo MobileNetV2 cho {NUM_CLASSES} classes...")
    # model = build_mobilenetv2_model(num_classes=NUM_CLASSES, pretrained=True)
    model = build_resnet50_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # ----------------------------------------------------------------
    # Chỉ truyền params có requires_grad=True vào Optimizer
    # Tránh lãng phí memory/compute tính gradient cho frozen layers
    # ----------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"🔧 Số param được optimize: {sum(p.numel() for p in trainable_params):,}")

    # ----------------------------------------------------------------
    # [YÊU CẦU 5] AdamW với weight_decay=0.01 (sửa từ 0.7 → 0.01)
    # - weight_decay=0.01: L2 regularization decoupled từ gradient update
    #   (Adam chuẩn implement weight decay sai → AdamW fix điều này)
    # - 0.7 là giá trị cũ sai lầm — quá lớn sẽ penalize weights cực mạnh
    #   khiến model không học được (gradient bị triệt tiêu bởi decay)
    # ----------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01   # [YÊU CẦU 5] Đã sửa từ 0.7 → 0.01
    )

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