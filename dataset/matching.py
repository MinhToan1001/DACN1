import pandas as pd
import os

# 1. Khai báo đường dẫn
csv_path = 'D:/hoctap/HK6/DACN1/dataset/dataset_cleaned.csv'  # File CSV bạn đã tải lên
images_folder = 'D:/hoctap/HK6/DACN1/images'
output_path = 'D:/hoctap/HK6/DACN1/dataset/dataset_matched.csv'

# 2. Đọc file CSV
df = pd.read_csv(csv_path)

# 3. Lấy danh sách tất cả các file ảnh hiện có trong thư mục (bao gồm cả thư mục con)
existing_images = set()
for root, dirs, files in os.walk(images_folder):
    for file in files:
        # Lấy tên file không có phần mở rộng (extension) để khớp với image_id
        file_name = os.path.splitext(file)[0]
        existing_images.add(file_name)

# 4. Lọc DataFrame: Chỉ giữ lại những dòng có image_id nằm trong danh sách ảnh thực tế
# Lưu ý: Đảm bảo image_id trong CSV khớp với tên file ảnh
df_matched = df[df['image_id'].isin(existing_images)]

# 5. Thông báo kết quả
original_count = len(df)
final_count = len(df_matched)
print(f"Số lượng dòng ban đầu: {original_count}")
print(f"Số lượng dòng sau khi lọc: {final_count}")
print(f"Đã xóa: {original_count - final_count} dòng không có ảnh tương ứng.")

# 6. Lưu file mới
df_matched.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"Đã lưu file sạch tại: {output_path}")