import os
import json

# Đường dẫn đến thư mục chứa ảnh lúc bạn train
# Nếu lúc train bạn để ảnh ở thư mục khác thì sửa lại tên ở đây nhé
DATA_DIR = "D:\HocTap\HK6\DACN1\images" 
MODEL_DIR = "D:\HocTap\HK6\DACN1\models"

def generate_class_names():
    if not os.path.exists(DATA_DIR):
        print(f"Lỗi: Không tìm thấy thư mục {DATA_DIR}. Hãy trỏ đúng vào thư mục ảnh của bạn.")
        return
        
    # 1. Quét thư mục ảnh, lấy tên các thư mục con và sắp xếp chuẩn A-Z
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    # 2. Tạo thư mục models nếu chưa có
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 3. Lưu ra file JSON
    out_path = os.path.join(MODEL_DIR, "class_names.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Đã trích xuất thành công {len(class_names)} loài!")
    print(f"✅ File đã được lưu tại: {out_path}")
    print("Mẫu 5 class đầu tiên để bạn kiểm tra:", class_names[:5])

if __name__ == "__main__":
    generate_class_names()