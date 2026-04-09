import torch
import torch.optim as optim

from src.models.model_resnet import build_resnet50_model
from src.utils.loss import FocalLoss  
from src.utils.train import ModelTrainer

# IMPORT THÊM DÒNG NÀY: Gọi pipeline từ file preprocess.py cùng thư mục
from src.data.preprocess import RareAnimalPipeline 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 20
    
    print("⏳ Đang chuẩn bị Dữ liệu qua Pipeline...")
    # BỎ COMMENT VÀ KHỞI TẠO PIPELINE
    pipeline = RareAnimalPipeline(
        data_dir="images", # Đảm bảo bạn thay bằng đường dẫn thư mục ảnh thật
        image_size=224,
        batch_size=32,
        use_focal_loss=True
    )
    result = pipeline.run()
    
    # Lấy dữ liệu trực tiếp từ RAM sang app.py
    train_loader = result['train_loader']
    val_loader = result['val_loader']
    
    # Dùng FocalLoss riêng
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    NUM_CLASSES = result['num_classes']

    print(f"🧠 Đang khởi tạo kiến trúc ResNet50 cho {NUM_CLASSES} classes...")
    model = build_resnet50_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # Khởi tạo Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("🚀 Bắt đầu huấn luyện...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion, # Truyền loss_fn đã lấy từ pipeline vào đây
        optimizer=optimizer,
        device=device
    )
    
    trainer.fit(epochs=EPOCHS)

if __name__ == "__main__":
    main()