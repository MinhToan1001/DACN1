import nbformat as nbf

nb = nbf.v4.new_notebook()

m1 = nbf.v4.new_markdown_cell("""# 🔥 Huấn Luyện EfficientNet-B3 Trên Kaggle
Notebook này thiết lập môi trường và chạy luồng Train cho EfficientNet-B3 sử dụng GPU miễn phí của Kaggle.

### Hướng dẫn chuẩn bị:
1. **Bật GPU:** Nhìn sang menu bên phải (Settings) -> Kéo xuống `Accelerator` -> Chọn **GPU T4 x2** (khuyên dùng) hoặc P100.
2. **Đưa mã nguồn & Data lên:** Gói toàn bộ nguyên dạng dự án `DACN1` (chứa `src`, `images`, `app.py`) thành file `.zip`, rồi upload lên Kaggle bằng nút **Add Data -> Upload** (Đặt tên là `wildlife-vietnam-dataset`).""")

c1 = nbf.v4.new_code_cell("!pip install grad-cam ttach")

m2 = nbf.v4.new_markdown_cell("""### Bước 1: Setup Môi trường
Mặc định khi bạn upload Zip lên Kaggle, bạn chỉ xem được data đọc (read-only). Code này tự động chuyển thư mục `src` sang ổ làm việc `/kaggle/working` để nó có thể load file config.""")

c2 = nbf.v4.new_code_cell("""import os
import shutil
import sys

# Chỉnh sửa nếu Kaggle đổi tên dataset của bạn (thường ở /kaggle/input/...)
KAGGLE_DATASET_PATH = '/kaggle/input/wildlife-vietnam-dataset'

KAGGLE_IMAGE_DIR = os.path.join(KAGGLE_DATASET_PATH, 'images')
src_source = os.path.join(KAGGLE_DATASET_PATH, 'src')
src_dest = '/kaggle/working/src'

# Copy source để import
if os.path.exists(src_source) and not os.path.exists(src_dest):
    shutil.copytree(src_source, src_dest)
    print("✅ Đã chép mã nguồn src thành output")

if '/kaggle/working' not in sys.path:
    sys.path.append('/kaggle/working')""")

m3 = nbf.v4.new_markdown_cell("### Bước 2: Bắt đầu Huấn Luyện")

c3 = nbf.v4.new_code_cell("""import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.model_efficientnet import build_efficientnet_b3_model
from src.utils.loss import FocalLoss
from src.utils.train import ModelTrainer
from src.data.preprocess import RareAnimalPipeline

def main_kaggle():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Thiết bị cấp phát: {device}")
    
    EPOCHS = 20
    NUM_CLASSES = 103 
    
    print("⏳ Data Pipeline setup...")
    pipeline = RareAnimalPipeline(
        data_dir=KAGGLE_IMAGE_DIR,
        image_size=224,
        batch_size=32,
        use_focal_loss=True
    )
    result = pipeline.run()

    train_loader = result['train_loader']
    val_loader   = result['val_loader']
    test_loader  = result['test_loader']
    class_names  = result['class_names']
    class_weights = result['class_weights'].to(device)
    
    criterion = FocalLoss(alpha=class_weights, gamma=2)

    print(f"🧠 Built EfficientNet-B3...")
    model = build_efficientnet_b3_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    optimizer = torch.optim.AdamW([
        {'params': model.features[6].parameters(), 'lr': 1e-5},
        {'params': model.features[7].parameters(), 'lr': 3e-5},
        {'params': model.features[8].parameters(), 'lr': 5e-5},
        {'params': model.classifier.parameters(),  'lr': 1e-4}
    ], weight_decay=0.01)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print("🚀 TRAINING...")
    trainer = ModelTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, criterion=criterion, optimizer=optimizer,
        scheduler=scheduler, device=device, class_names=class_names
    )

    trainer.fit(epochs=EPOCHS)
    
    # Save Output
    torch.save(model.state_dict(), '/kaggle/working/best_efficientnet_b3.pt')
    print("✅ Xong! Check Output File trên Kaggle ở góc phải.")

if __name__ == '__main__':
    main_kaggle()""")

nb['cells'] = [m1, c1, m2, c2, m3, c3]

import os
os.makedirs("notebook", exist_ok=True)
with open('notebook/Train_EfficientNet_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook generated successfully!")
