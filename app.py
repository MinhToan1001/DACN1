import torch
import torch.optim as optim

# Import các module đã tách biệt
from src.models.model_resnet import build_resnet50_model
from src.utils.loss import FocalLoss
from src.utils.train import ModelTrainer

# Import Pipeline Data (Giả sử bạn lấy từ preprocess.py của bạn)
# from src.data.preprocess import RareAnimalPipeline 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 103 # Sửa thành số lượng loài thực tế của bạn
    EPOCHS = 20
    
    # 1. Chuẩn bị Dữ liệu (Load từ tiền xử lý)
    print("⏳ Đang chuẩn bị Dữ liệu...")
    # pipeline = RareAnimalPipeline(...)
    # result = pipeline.run()
    # train_loader = result['train_loader']
    # val_loader = result['val_loader']
    
    # (Mock data loaders để code không báo lỗi khi bạn test)
    train_loader = [] 
    val_loader = []

    # 2. Khởi tạo Mô hình (Gọi từ file model.py)
    print("🧠 Đang khởi tạo kiến trúc Mô hình...")
    model = build_resnet50_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # 3. Khởi tạo Hàm Loss và Optimizer (Gọi từ file loss.py)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 4. Truyền vào Trainer và bắt đầu huấn luyện (Gọi từ file train.py)
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    trainer.fit(epochs=EPOCHS)

if __name__ == "__main__":
    main()