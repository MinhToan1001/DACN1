import pandas as pd
import os

# --- CẤU HÌNH ---
# Danh sách 100 loài khớp với model
CLASS_NAMES = [
    'acinonyx_jubatus', 'addax_nasomaculatus', 'aedes_aegypti', 'ailuropoda_melanoleuca', 'amatitlania_nigrofasciata',
    'amblyrhynchus_cristatus', 'anas_platyrhynchos_domesticus', 'anser_anser_domesticus', 'apis_mellifera', 'balaenoptera_musculus',
    'betta_splendens', 'bos_taurus', 'bubalus_bubalis', 'camelus_ferus', 'canis_latrans',
    'canis_lupus_familiaris', 'capra_aegagrus_hircus', 'carassius_auratus', 'chelonoidis_nigra', 'chromobotia_macracanthus',
    'ciconia_boyciana', 'columba_livia_domestica', 'corvus_brachyrhynchos', 'corydoras_paleatus', 'crocodylus_siamensis',
    'cuon_alpinus', 'cyprinus_carpio', 'danio_rerio', 'daubentonia_madagascariensis', 'dermochelys_coriacea',
    'diceros_bicornis', 'elephas_maximus', 'equus_ferus_caballus', 'equus_ferus_przewalskii', 'eretmochelys_imbricata',
    'felis_catus', 'gallus_gallus_domesticus', 'gavialis_gangeticus', 'gazella_leptoceros', 'gekko_gecko',
    'gorilla_beringei', 'gorilla_gorilla', 'grus_japonensis', 'gymnogyps_californianus', 'hemidactylus_frenatus',
    'hypostomus_plecostomus', 'indri_indri', 'leucopsar_rothschildi', 'lonchura_striata_domestica', 'lynx_pardinus',
    'manis_pentadactyla', 'melopsittacus_undulatus', 'mesocricetus_auratus', 'mus_musculus', 'musca_domestica',
    'neofelis_nebulosa', 'nipponia_nippon', 'nomascus_nasutus', 'nycticebus_pygmaeus', 'nymphicus_hollandicus',
    'oreochromis_niloticus', 'oryctolagus_cuniculus', 'oryx_dammah', 'ovis_aries', 'pan_paniscus',
    'pan_troglodytes', 'panthera_leo_persica', 'panthera_pardus_orientalis', 'panthera_tigris_corbetti', 'panthera_uncia',
    'paracheirodon_innesi', 'passer_domesticus', 'periplaneta_americana', 'phocoena_sinus', 'pica_pica',
    'pithecophaga_jefferyi', 'poecilia_reticulata', 'pongo_abelii', 'procyon_lotor', 'propithecus_tattersalli',
    'pseudoryx_nghetinhensis', 'pterophyllum_scalare', 'puntius_tetrazona', 'rafetus_swinhoei', 'rattus_norvegicus',
    'rhinoceros_sondaicus', 'rhinopithecus_avunculus', 'saiga_tatarica', 'sciurus_carolinensis', 'serinus_canaria',
    'strigops_habroptila', 'sturnus_vulgaris', 'sus_scrofa_domesticus', 'taeniopygia_guttata', 'trachypithecus_poliocephalus',
    'trichogaster_lalius', 'turdus_migratorius', 'ursus_thibetanus', 'varecia_variegata', 'xiphophorus_hellerii'
]

# --- HÀM PHÂN LOẠI CHI TIẾT ---
def get_detailed_legal_info(folder_name):
    """
    Trả về thông tin pháp lý VÀ SINH HỌC đầy đủ.
    """
    
    name = folder_name.lower()
    
    # 1. NHÓM VẬT NUÔI / GIA SÚC / GIA CẦM
    pets_livestock = [
        'canis_lupus_familiaris', 'felis_catus', 'bos_taurus', 'bubalus_bubalis', 
        'gallus_gallus_domesticus', 'anas_platyrhynchos_domesticus', 'anser_anser_domesticus',
        'sus_scrofa_domesticus', 'ovis_aries', 'capra_aegagrus_hircus', 'equus_ferus_caballus',
        'oryctolagus_cuniculus', 'mesocricetus_auratus', 'columba_livia_domestica', 'serinus_canaria'
    ]
    
    if name in pets_livestock:
        return {
            "vietnamese_name": name.replace("_", " ").title(),
            "legal_group": "Vật nuôi / Gia súc / Gia cầm",
            "decree": "Luật Chăn nuôi 2018 & Luật Thú y",
            "penalty_warning": "Không bị phạt tù. Xử phạt hành chính nếu thả rông, không tiêm phòng hoặc gây ô nhiễm môi trường.",
            "farming_advice": "ĐƯỢC PHÉP NUÔI TỰ DO. Khuyến nghị: Tiêm phòng dại/dịch bệnh định kỳ, đăng ký kê khai chăn nuôi với xã/phường.",
            "status_code": "success",
            # THÔNG TIN SINH HỌC CỤ THỂ
            "habitat": "Sống cùng con người, trong nhà, trang trại hoặc khu chăn thả.",
            "diet": "Thức ăn hỗn hợp công nghiệp, thực phẩm thừa, cỏ, ngũ cốc.",
            "behavior": "Đã được thuần hóa, thân thiện, phụ thuộc vào con người.",
            "description": "Loài vật gắn bó với đời sống con người, phục vụ nhu cầu thực phẩm, giải trí hoặc canh gác."
        }

    # 2. NHÓM CÔN TRÙNG / GÂY HẠI
    pests_insects = [
        'aedes_aegypti', 'musca_domestica', 'periplaneta_americana', 
        'rattus_norvegicus', 'mus_musculus'
    ]
    
    if name in pests_insects:
        return {
            "vietnamese_name": name.replace("_", " ").title(),
            "legal_group": "Động vật gây hại / Côn trùng",
            "decree": "Quy định Y tế & Vệ sinh môi trường",
            "penalty_warning": "Không bảo tồn. Cần kiểm soát để tránh lây lan dịch bệnh truyền nhiễm.",
            "farming_advice": "KHÔNG KHUYẾN KHÍCH NUÔI (trừ mục đích nghiên cứu). Cần tiêu diệt hoặc kiểm soát sự sinh sôi.",
            "status_code": "warning",
            # THÔNG TIN SINH HỌC CỤ THỂ
            "habitat": "Cống rãnh, nơi ẩm thấp, khu dân cư, rác thải.",
            "diet": "Tạp ăn, rác thải, máu động vật (đối với muỗi).",
            "behavior": "Sinh sản rất nhanh, hoạt động mạnh về đêm hoặc nơi tối tăm.",
            "description": "Nhóm động vật thường mang mầm bệnh nguy hiểm cho con người."
        }

    # 3. NHÓM CÁ CẢNH THÔNG THƯỜNG
    aquarium_fish = [
        'betta_splendens', 'carassius_auratus', 'poecilia_reticulata', 'paracheirodon_innesi',
        'pterophyllum_scalare', 'xiphophorus_hellerii', 'trichogaster_lalius', 'corydoras_paleatus',
        'hypostomus_plecostomus', 'chromobotia_macracanthus', 'puntius_tetrazona', 'danio_rerio',
        'amatitlania_nigrofasciata', 'oreochromis_niloticus', 'cyprinus_carpio'
    ]
    
    if name in aquarium_fish:
        return {
            "vietnamese_name": name.replace("_", " ").title(),
            "legal_group": "Thủy sản / Cá cảnh thông thường",
            "decree": "Luật Thủy sản 2017",
            "penalty_warning": "Phạt hành chính nếu thả loài ngoại lai xâm hại (như cá lau kính, rô phi) ra môi trường tự nhiên.",
            "farming_advice": "ĐƯỢC PHÉP NUÔI THƯƠNG MẠI/LÀM CẢNH. Cần hệ thống lọc nước tốt và tuân thủ quy định kiểm dịch.",
            "status_code": "success",
            # THÔNG TIN SINH HỌC CỤ THỂ
            "habitat": "Môi trường nước ngọt, bể thủy sinh, ao hồ.",
            "diet": "Cám viên, trùn chỉ, bo bo, rong rêu.",
            "behavior": "Bơi theo đàn hoặc đơn lẻ, thích nghi tốt với môi trường nhân tạo.",
            "description": "Các loài cá phổ biến được nuôi làm cảnh hoặc lấy thịt, có giá trị kinh tế."
        }

    # 4. NHÓM CỰC KỲ QUÝ HIẾM (NHÓM IB - NGHỊ ĐỊNH 06/2019)
    critically_endangered = [
        'panthera', 'pseudoryx', 'rhinoceros', 'elephas', 'gorilla', 'pongo', 'pan_troglodytes', 'pan_paniscus',
        'manis', 'cuon_alpinus', 'ursus', 'gavialis', 'rafetus', 'nomascus', 'pygathrix', 'rhinopithecus',
        'trachypithecus', 'nycticebus', 'hylobates', 'catopuma', 'pardofelis', 'neofelis', 'indri', 'propithecus',
        'varecia', 'daubentonia', 'balaenoptera', 'vaquita', 'phocoena', 'pithecophaga', 'gymnogyps', 'strigops',
        'leucopsar', 'nipponia', 'grus_japonensis', 'ciconia_boyciana', 'saiga', 'addax', 'oryx', 'gazella',
        'lynx_pardinus', 'acinonyx', 'ailuropoda', 'eretmochelys', 'dermochelys', 'chelonoidis', 'amblyrhynchus'
    ]
    
    for key in critically_endangered:
        if key in name:
            return {
                "vietnamese_name": name.replace("_", " ").title(),
                "legal_group": "NHÓM IB (Nghiêm cấm tuyệt đối)",
                "decree": "Nghị định 06/2019/NĐ-CP & Điều 244 BLHS",
                "penalty_warning": "CẢNH BÁO ĐỎ: Phạt tiền 500tr - 2 tỷ đồng hoặc TÙ GIAM 1-15 năm nếu săn bắt, nuôi nhốt trái phép.",
                "farming_advice": "TUYỆT ĐỐI CẤM NUÔI THƯƠNG MẠI. Chỉ phục vụ bảo tồn/nghiên cứu với giấy phép đặc biệt cấp Bộ.",
                "status_code": "danger",
                # THÔNG TIN SINH HỌC CỤ THỂ
                "habitat": "Rừng nguyên sinh, khu bảo tồn thiên nhiên, vùng núi cao hoặc biển sâu.",
                "diet": "Đa dạng tùy loài (Thịt, thực vật, trái cây rừng).",
                "behavior": "Hoang dã, hung dữ hoặc nhút nhát, khó sinh sản trong môi trường nhân tạo.",
                "description": "Loài đặc biệt quý hiếm, có nguy cơ tuyệt chủng rất cao, cần được bảo vệ nghiêm ngặt."
            }

    # 5. NHÓM ĐVHD CẦN BẢO VỆ (NHÓM IIB)
    return {
        "vietnamese_name": name.replace("_", " ").title(),
        "legal_group": "Nhóm IIB / ĐVHD Thông thường (Hạn chế)",
        "decree": "Nghị định 06/2019/NĐ-CP & Nghị định 84/2021",
        "penalty_warning": "Phạt tiền 50 - 300 triệu hoặc cải tạo không giam giữ nếu tàng trữ không phép.",
        "farming_advice": "NUÔI CÓ ĐIỀU KIỆN. Bắt buộc: 1. Đăng ký mã số trại nuôi. 2. Nguồn gốc con giống hợp pháp. 3. Sổ theo dõi.",
        "status_code": "warning",
        # THÔNG TIN SINH HỌC CỤ THỂ
        "habitat": "Rừng thứ sinh, trảng cỏ, đất ngập nước.",
        "diet": "Côn trùng, hạt, trái cây, động vật nhỏ.",
        "behavior": "Thích nghi được với môi trường bán hoang dã.",
        "description": "Động vật hoang dã thông thường hoặc ít nguy cấp, có thể gây nuôi thương mại nếu tuân thủ pháp luật."
    }

# --- TẠO DỮ LIỆU ---
data = []
for folder_name in CLASS_NAMES:
    row = get_detailed_legal_info(folder_name)
    row['folder_name'] = folder_name
    data.append(row)

# Tạo DataFrame và lưu ra CSV
df = pd.DataFrame(data)
os.makedirs('animal_dataset', exist_ok=True)
output_path = 'animal_dataset/legal_database.csv'
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"✅ Đã cập nhật file cơ sở dữ liệu pháp lý & sinh học ĐẦY ĐỦ tại: {output_path}")
print("👉 File mới đã có thêm cột: Môi trường sống, Thức ăn, Tập tính.")