import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def sync_images_with_csv(csv_path, images_dir):
    print("⏳ Đang đọc CSV...")

    df_cleaned = pd.read_csv(csv_path, encoding='latin-1')

    # Lấy tên file KHÔNG có extension
    valid_filenames = set(
        Path(str(p)).stem for p in df_cleaned['image_id'].dropna()
    )

    print(f"📋 CSV giữ lại: {len(valid_filenames)} ảnh")

    images_path = Path(images_dir)
    all_images = list(images_path.rglob("*.[jp][pn]*[g]"))

    print(f"📁 Thư mục có: {len(all_images)} ảnh")

    deleted_count = 0
    kept_count = 0

    print("🗑️ Đang xử lý...")

    for img_path in tqdm(all_images):
        img_name = img_path.stem  # 🔥 FIX CHÍNH Ở ĐÂY

        if img_name not in valid_filenames:
            try:
                img_path.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Lỗi xóa {img_path}: {e}")
        else:
            kept_count += 1

    print("\n🎉 DONE")
    print(f"✅ Giữ: {kept_count}")
    print(f"❌ Xóa: {deleted_count}")
# ====================== THỰC THI ======================
if __name__ == "__main__":
    csv_path = r"D:/HocTap/HK6/DACN1/dataset/dataset_cleaned.csv"
    images_dir = r"D:/HocTap/HK6/DACN1/images"
    
    sync_images_with_csv(csv_path, images_dir)