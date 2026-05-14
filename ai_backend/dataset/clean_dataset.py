import pandas as pd

# 1. Đọc file dữ liệu của bạn
df = pd.read_csv("D:/hoctap/HK6/DACN1/dataset/dataset_final.csv")
print(f"📊 Tổng số ảnh ban đầu: {len(df)}")

# 2. Xóa các ảnh rò rỉ (cùng loài, cùng tọa độ, cùng thời gian)
# keep='first' nghĩa là trong 5 ảnh giống nhau, ta chỉ giữ lại ảnh đầu tiên
df_cleaned = df.drop_duplicates(
    subset=['scientific_name', 'latitude', 'longitude', 'observed_time'], 
    keep='first'
)

# 3. Thống kê kết quả
removed_count = len(df) - len(df_cleaned)
print(f"🧹 Đã loại bỏ: {removed_count} ảnh chụp liên tiếp (rò rỉ dữ liệu)!")
print(f"✅ Tổng số ảnh sạch còn lại: {len(df_cleaned)}")

# 4. Lưu ra file CSV mới để dùng cho huấn luyện
df_cleaned.to_csv("D:/hoctap/HK6/DACN1/dataset/dataset_cleaned.csv", index=False)
print("💾 Đã lưu file 'dataset_cleaned.csv'. Hãy dùng file này để train model!")