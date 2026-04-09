import pandas as pd
import json
import wikipedia

# Cài đặt ngôn ngữ Wikipedia là tiếng Việt
wikipedia.set_lang("vi")

# 1. Đọc dữ liệu từ CSV
try:
    df = pd.read_csv('D:/hoctap/HK6/DACN1/dataset/dataset_final.csv')
    unique_species = df[['scientific_name', 'common_name_vn', 'legal_group']].drop_duplicates()
except FileNotFoundError:
    print("Không tìm thấy file dataset_final.csv. Vui lòng kiểm tra lại đường dẫn.")
    exit()

# 2. Xây dựng bộ luật CHI TIẾT TỔNG HỢP (Nhóm IB, Nhóm IIB và Thông thường)
# 2. Xây dựng bộ luật CHI TIẾT TỔNG HỢP (Cập nhật Nhóm IIB theo file 2B.txt)
laws_config = {
    "IB": {
        "group_name": "Nhóm IB - Động vật rừng nguy cấp, quý, hiếm",
        "criminal_penalties": {
            "Khung_1_Phat_Tien_1_den_4_Ty_Hoac_Tu_1_den_5_Nam": [
                "Săn bắt, giết, nuôi, nhốt, vận chuyển, buôn bán trái phép động vật thuộc Danh mục loài nguy cấp, quý, hiếm được ưu tiên bảo vệ[cite: 2].",
                "Tàng trữ, vận chuyển, buôn bán trái phép cá thể, bộ phận cơ thể không thể tách rời sự sống hoặc sản phẩm của chúng[cite: 3].",
                "Tàng trữ, vận chuyển, buôn bán trái phép ngà voi có khối lượng từ 02 kilôgam đến dưới 20 kilôgam; sừng tê giác có khối lượng từ 50 gam đến dưới 01 kilôgam[cite: 4, 5].",
                "Động vật Nhóm IB hoặc Phụ lục I CITES (không ưu tiên bảo vệ): Số lượng từ 03 đến 07 cá thể lớp thú, từ 07 đến 10 cá thể lớp chim, bò sát hoặc từ 10 đến 15 cá thể động vật lớp khác[cite: 6].",
                "Tàng trữ, vận chuyển, buôn bán bộ phận cơ thể của các động vật Nhóm IB theo số lượng nêu trên[cite: 7].",
                "Vi phạm dưới mức số lượng trên nhưng đã bị xử phạt vi phạm hành chính hoặc đã bị kết án, chưa được xóa án tích mà còn vi phạm[cite: 8]."
            ],
            "Khung_2_Phat_Tu_5_den_10_Nam": [
                "Phạm tội có tổ chức; Lợi dụng chức vụ, quyền hạn, danh nghĩa cơ quan; Sử dụng công cụ/phương tiện cấm; Săn bắt trong khu vực/thời gian cấm; Buôn bán, vận chuyển qua biên giới; Tái phạm nguy hiểm[cite: 9, 10, 11, 12, 13, 18].",
                "Động vật ưu tiên bảo vệ: Từ 07 đến 10 cá thể lớp thú, từ 07 đến 10 cá thể lớp chim, lớp bò sát hoặc từ 10 đến 15 cá thể lớp khác[cite: 13].",
                "Động vật Nhóm IB khác: Từ 08 đến 11 cá thể thuộc lớp thú, từ 11 đến 15 cá thể lớp chim, bò sát hoặc từ 16 đến 20 cá thể động vật thuộc các lớp khác[cite: 14].",
                "Từ 01 đến 02 cá thể voi, tê giác; từ 03 đến 05 cá thể gấu, hổ (hoặc bộ phận cơ thể không thể tách rời)[cite: 15, 16].",
                "Ngà voi có khối lượng từ 20 kilôgam đến dưới 90 kilôgam; sừng tê giác có khối lượng từ 01 kilôgam đến dưới 09 kilôgam[cite: 17, 18]."
            ],
            "Khung_3_Phat_Tu_10_den_15_Nam": [
                "Động vật ưu tiên bảo vệ: Từ 08 cá thể lớp thú trở lên, 11 cá thể lớp chim, lớp bò sát trở lên hoặc 16 cá thể lớp khác trở lên[cite: 19].",
                "Động vật Nhóm IB khác: 12 cá thể lớp thú trở lên, 16 cá thể lớp chim, bò sát trở lên hoặc 21 cá thể động vật trở lên thuộc các lớp khác[cite: 20].",
                "Từ 03 cá thể voi, tê giác trở lên; 06 cá thể gấu, hổ trở lên (hoặc bộ phận cơ thể)[cite: 21, 22].",
                "Ngà voi có khối lượng 90 kilôgam trở lên; sừng tê giác có khối lượng 09 kilôgam trở lên[cite: 23]."
            ],
            "Hinh_Phat_Bo_Sung_Ca_Nhan": [
                "Phạt tiền từ 50.000.000 đồng đến 200.000.000 đồng, cấm đảm nhiệm chức vụ, cấm hành nghề hoặc làm công việc nhất định từ 01 năm đến 05 năm[cite: 24]."
            ],
            "Xu_Phat_Phap_Nhan_Thuong_Mai": [
                "Phạm tội thuộc Khung 1: Phạt tiền từ 1.000.000.000 đồng đến 5.000.000.000 đồng[cite: 25].",
                "Phạm tội thuộc Khung 2: Phạt tiền từ 5.000.000.000 đồng đến 10.000.000.000 đồng[cite: 26].",
                "Phạm tội thuộc Khung 3: Phạt tiền từ 10.000.000.000 đồng đến 15.000.000.000 đồng hoặc đình chỉ hoạt động từ 06 tháng đến 03 năm[cite: 27].",
                "Đình chỉ hoạt động vĩnh viễn (thuộc Điều 79 BLHS 2015)[cite: 28].",
                "Hình phạt bổ sung: Phạt tiền từ 300.000.000 đồng đến 600.000.000 đồng, cấm kinh doanh, cấm hoạt động/huy động vốn từ 01 năm đến 03 năm[cite: 29]."
            ]
        }
    },
    "IIB": {
        "group_name": "Nhóm IIB - Động vật rừng nguy cấp, quý, hiếm (Hạn chế khai thác)",
        "administrative_penalties": {
            "Khung_1_Phat_1_den_5_Trieu": "Động vật Nhóm IIB trị giá dưới 3.000.000 đồng[cite: 2].",
            "Khung_1a_Phat_5_den_10_Trieu": "Động vật Nhóm IIB trị giá từ 3.000.000 đồng đến dưới 5.000.000 đồng[cite: 4].",
            "Khung_2_Phat_10_den_25_Trieu": "Động vật Nhóm IIB trị giá từ 5.000.000 đồng đến dưới 10.000.000 đồng[cite: 6].",
            "Khung_3_Phat_25_den_50_Trieu": "Động vật Nhóm IIB trị giá từ 10.000.000 đồng đến dưới 20.000.000 đồng[cite: 8].",
            "Khung_4_Phat_50_den_80_Trieu": "Động vật Nhóm IIB trị giá từ 20.000.000 đồng đến dưới 35.000.000 đồng[cite: 10].",
            "Khung_5_Phat_80_den_110_Trieu": "Động vật Nhóm IIB trị giá từ 35.000.000 đồng đến dưới 50.000.000 đồng[cite: 12].",
            "Khung_6_Phat_110_den_140_Trieu": "Động vật Nhóm IIB trị giá từ 50.000.000 đồng đến dưới 65.000.000 đồng[cite: 14].",
            "Khung_7_Phat_140_den_170_Trieu": "Động vật Nhóm IIB trị giá từ 65.000.000 đồng đến dưới 80.000.000 đồng[cite: 16].",
            "Khung_8_Phat_170_den_210_Trieu": "Động vật Nhóm IIB trị giá từ 80.000.000 đồng đến dưới 95.000.000 đồng[cite: 18].",
            "Khung_9_Phat_210_den_240_Trieu": "Động vật Nhóm IIB trị giá từ 95.000.000 đồng đến dưới 110.000.000 đồng[cite: 20].",
            "Khung_10_Phat_240_den_270_Trieu": "Động vật Nhóm IIB trị giá từ 110.000.000 đồng đến dưới 125.000.000 đồng[cite: 22].",
            "Khung_11_Phat_270_den_300_Trieu": "Động vật Nhóm IIB trị giá từ 125.000.000 đồng đến dưới 150.000.000 đồng[cite: 24]."
        },
        "additional_penalties": [
            "Tịch thu tang vật, dụng cụ, công cụ vi phạm đối với tất cả các khung hình phạt nêu trên[cite: 28].",
            "Tịch thu phương tiện vi phạm đối với hành vi bị xử phạt từ khung (3) trở lên (mức phạt từ 25 triệu đồng đến 300 triệu đồng)[cite: 29]."
        ],
        "remedial_measures": [
            "Buộc thực hiện biện pháp khắc phục tình trạng ô nhiễm môi trường, lây lan dịch bệnh[cite: 30].",
            "Buộc tiêu hủy hàng hóa, vật phẩm gây hại cho sức khỏe con người, vật nuôi, cây trồng và môi trường[cite: 31]."
        ]
    },
    "THONG_THUONG": {
        "group_name": "Động vật rừng thông thường",
        "administrative_penalties": {
            "Khung_1_Phat_1_den_5_Trieu": "Trị giá tang vật dưới 5.000.000 đồng[cite: 1].",
            "Khung_1a_Phat_5_den_10_Trieu": "Trị giá tang vật từ 5.000.000 đồng đến dưới 10.000.000 đồng[cite: 3].",
            "Khung_2_Phat_10_den_25_Trieu": "Trị giá tang vật từ 10.000.000 đồng đến dưới 20.000.000 đồng[cite: 5].",
            "Khung_3_Phat_25_den_50_Trieu": "Trị giá tang vật từ 20.000.000 đồng đến dưới 40.000.000 đồng[cite: 7].",
            "Khung_4_Phat_50_den_80_Trieu": "Trị giá tang vật từ 40.000.000 đồng đến dưới 70.000.000 đồng[cite: 9].",
            "Khung_5_Phat_80_den_110_Trieu": "Trị giá tang vật từ 70.000.000 đồng đến dưới 100.000.000 đồng[cite: 11].",
            "Khung_6_Phat_110_den_140_Trieu": "Trị giá tang vật từ 100.000.000 đồng đến dưới 130.000.000 đồng[cite: 13].",
            "Khung_7_Phat_140_den_170_Trieu": "Trị giá tang vật từ 130.000.000 đồng đến dưới 160.000.000 đồng[cite: 15].",
            "Khung_8_Phat_170_den_210_Trieu": "Trị giá tang vật từ 160.000.000 đồng đến dưới 190.000.000 đồng[cite: 17].",
            "Khung_9_Phat_210_den_240_Trieu": "Trị giá tang vật từ 190.000.000 đồng đến dưới 220.000.000 đồng[cite: 19].",
            "Khung_10_Phat_240_den_270_Trieu": "Trị giá tang vật từ 220.000.000 đồng đến dưới 250.000.000 đồng[cite: 21].",
            "Khung_11_Phat_270_den_300_Trieu": "Trị giá tang vật từ 250.000.000 đồng đến dưới 300.000.000 đồng[cite: 23]."
        },
        "additional_and_remedial": {
            "hinh_thuc_bo_sung": "Tịch thu tang vật, dụng cụ, công cụ[cite: 28]. Tịch thu phương tiện vi phạm đối với hành vi bị phạt từ mức 25.000.000 đồng trở lên[cite: 29].",
            "khac_phuc_hau_qua": "Buộc thực hiện biện pháp khắc phục tình trạng ô nhiễm môi trường, lây lan dịch bệnh [cite: 30]; buộc tiêu hủy hàng hóa, vật phẩm gây hại[cite: 31]."
        }
    },
    "1": {
        "group_name": "Danh mục loài thủy sản nguy cấp, quý, hiếm (Nhóm I)",
        "administrative_penalties": {
            "Vi_pham_khai_thac_trong_vung_thoi_gian_cam": [
                "Phạt tiền đến 100.000.000 đồng đối với hành vi khai thác thủy sản trong vùng cấm, thời gian cấm (Điều 7 Nghị định 42/2019/NĐ-CP).",
                "Hình phạt bổ sung: Tịch thu ngư cụ khai thác thủy sản.",
                "Biện pháp khắc phục: Buộc thả thủy sản còn sống trở lại môi trường; buộc chuyển giao số thủy sản thuộc Nhóm I đã chết cho cơ quan có thẩm quyền xử lý."
            ],
            "Vi_pham_khong_giay_phep_hoac_ngoai_vung_bien": [
                "Phạt tiền đến 1.000.000.000 đồng đối với chủ tàu cá khai thác không có giấy phép hoặc khai thác trái phép trong vùng biển quốc gia/vùng lãnh thổ khác (Điều 20, 23 Nghị định 42/2019/NĐ-CP).",
                "Hình phạt bổ sung: Tịch thu thủy sản khai thác, tịch thu ngư cụ, tịch thu tàu cá, tước quyền sử dụng văn bằng, chứng chỉ thuyền trưởng.",
                "Biện pháp khắc phục: Buộc chủ tàu cá chi trả toàn bộ kinh phí đưa ngư dân bị nước ngoài bắt giữ về nước và các chi phí liên quan."
            ]
        },
        "criminal_penalties": {
            "Truy_cuu_trach_nhiem_hinh_su": [
                "Tùy theo mức độ vi phạm khai thác thủy sản bất hợp pháp (khoản 2 Điều 60 Luật Thủy sản 2017), tổ chức cá nhân có thể bị truy cứu trách nhiệm hình sự (thường chiếu theo Điều 244 Bộ luật Hình sự về tội vi phạm quy định bảo vệ động vật nguy cấp, quý, hiếm)."
            ]
        }
    }
}
rulebase = {}
print("Đang tạo Rulebase tích hợp Wikipedia và Luật...")

for index, row in unique_species.iterrows():
    sci_name = str(row['scientific_name']).strip()
    common_name = str(row['common_name_vn']).strip()
    
    if pd.isna(row['scientific_name']):
        continue
        
    group = str(row['legal_group']).strip().upper()
    
    # Xử lý gán luật
    if group == "IB":
        legal_info = laws_config["IB"]
    elif group == "IIB":
        legal_info = laws_config["IIB"]
    else:
        legal_info = laws_config["THONG_THUONG"]

    # Trích xuất Wikipedia
    wiki_summary = "Chưa có dữ liệu sinh học cụ thể."
    try:
        search_query = sci_name if pd.notna(row['scientific_name']) else common_name
        wiki_summary = wikipedia.summary(search_query, sentences=3)
        print(f"✅ Xong: {common_name}")
    except:
        pass

    # Đưa vào dictionary chính
    rulebase[sci_name] = {
        "common_name": common_name if common_name != 'nan' else sci_name,
        "scientific_name": sci_name,
        "legal_group": group if group in ["IB", "IIB"] else "THÔNG THƯỜNG",
        "biological_info": wiki_summary,
        "legal_advice": legal_info
    }

# Lưu file JSON
with open('D:/hoctap/HK6/DACN1/rules/animal_rulebase.json', 'w', encoding='utf-8') as f:
    json.dump(rulebase, f, ensure_ascii=False, indent=4)

print("\n🎉 Tạo thành công animal_rulebase.json!")