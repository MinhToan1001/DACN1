import pandas as pd

# 1. Đọc file dataset_cleaned.csv gốc
dataset_path = "D:/hoctap/HK6/DACN1/dataset/dataset_cleaned.csv"
df_dataset = pd.read_csv(dataset_path)

# 2. Thống kê: Đếm số lượng ảnh theo từng tên khoa học
# Mỗi dòng là 1 ảnh, nên value_counts() sẽ trả về số lượng ảnh của từng loài
thong_ke = df_dataset['scientific_name'].value_counts().reset_index()

# 3. Đổi tên cột cho khớp với logic trong file extract_species_info.py của bạn
# Trong extract_species_info.py bạn dùng cột "Tên loài (Thư mục)" và "Số lượng ảnh"
thong_ke.columns = ['Tên loài (Thư mục)', 'Số lượng ảnh']

# In kết quả ra màn hình (chỉ lấy tên khoa học và số lượng ảnh như bạn yêu cầu)
print("=== THỐNG KÊ SỐ LƯỢNG ẢNH THEO TÊN KHOA HỌC ===")
print(thong_ke)

# 4. Lưu lại thành file CSV thống kê mới
output_path = "D:/hoctap/HK6/DACN1/dataset/thong_ke_du_lieu_moi.csv"
thong_ke.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n[Thành công] Đã tạo lại file thống kê tại: {output_path}")