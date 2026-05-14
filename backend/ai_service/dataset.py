import os
import shutil
import random
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# --- CẤU HÌNH ---
BASE_DIR = "animal_dataset"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
CSV_FILE = os.path.join(BASE_DIR, "animal_labels.csv")

# Số lượng ảnh giả lập muốn tạo cho MỖI loài
IMAGES_PER_SPECIES = 5 

# --- DANH SÁCH 50 LOÀI QUÝ HIẾM (SCIENTIFIC NAMES - Label 1) ---
# Bao gồm các loài nguy cấp (CR), nguy cấp (EN) và sắp nguy cấp (VU)
rare_species_scientific = [
    "Pseudoryx nghetinhensis",    # Sao La
    "Rhinoceros sondaicus",       # Tê giác Java
    "Panthera tigris corbetti",   # Hổ Đông Dương
    "Rhinopithecus avunculus",    # Voọc mũi hếch
    "Panthera pardus orientalis", # Báo Amur
    "Rafetus swinhoei",           # Rùa Hoàn Kiếm
    "Elephas maximus",            # Voi châu Á
    "Ailuropoda melanoleuca",     # Gấu trúc lớn
    "Manis pentadactyla",         # Tê tê vàng
    "Pithecophaga jefferyi",      # Đại bàng Philippines
    "Phocoena sinus",             # Cá heo Vaquita
    "Gorilla beringei",           # Khỉ đột núi
    "Crocodylus siamensis",       # Cá sấu Xiêm
    "Ciconia boyciana",           # Hạc trắng Á Đông
    "Panthera uncia",             # Báo tuyết
    "Cuon alpinus",               # Sói đỏ
    "Nycticebus pygmaeus",        # Cu li nhỏ
    "Ursus thibetanus",           # Gấu ngựa
    "Trachypithecus poliocephalus", # Voọc Cát Bà
    "Nomascus nasutus",           # Vượn Cao Vít
    "Eretmochelys imbricata",     # Đồi mồi
    "Dermochelys coriacea",       # Rùa da
    "Pan troglodytes",            # Tinh tinh
    "Pongo abelii",               # Đười ươi Sumatra
    "Diceros bicornis",           # Tê giác đen
    "Balaenoptera musculus",      # Cá voi xanh
    "Equus ferus przewalskii",    # Ngựa Przewalski
    "Camelus ferus",              # Lạc đà hai bướu hoang dã
    "Grus japonensis",            # Sếu Nhật Bản
    "Nipponia nippon",            # Cò quăm mào Nhật Bản
    "Panthera leo persica",       # Sư tử châu Á
    "Acinonyx jubatus",           # Báo săn
    "Lynx pardinus",              # Linh miêu Iberia
    "Addax nasomaculatus",        # Linh dương sừng xoắn
    "Gazella leptoceros",         # Linh dương sừng thanh
    "Oryx dammah",                # Linh dương sừng kiếm
    "Gorilla gorilla",            # Khỉ đột đất thấp phương Tây
    "Pan paniscus",               # Tinh tinh lùn (Bonobo)
    "Indri indri",                # Vượn cáo Indri
    "Propithecus tattersalli",    # Vượn cáo Sifaka
    "Varecia variegata",          # Vượn cáo râu choàng cổ
    "Daubentonia madagascariensis", # Khỉ aye-aye
    "Amblyrhynchus cristatus",    # Cự đà biển
    "Chelonoidis nigra",          # Rùa khổng lồ Galápagos
    "Strigops habroptila",        # Vẹt Kakapo
    "Gymnogyps californianus",    # Kền kền California
    "Leucopsar rothschildi",      # Sáo Bali
    "Gavialis gangeticus",        # Cá sấu sông Hằng
    "Neofelis nebulosa",          # Báo gấm
    "Saiga tatarica"              # Linh dương Saiga
]

# --- DANH SÁCH 50 LOÀI PHỔ BIẾN (SCIENTIFIC NAMES - Label 0) ---
# Bao gồm vật nuôi và động vật hoang dã phổ biến
common_species_scientific = [
    "Canis lupus familiaris",     # Chó nhà
    "Felis catus",                # Mèo nhà
    "Gallus gallus domesticus",   # Gà nhà
    "Anas platyrhynchos domesticus", # Vịt nhà
    "Bos taurus",                 # Bò
    "Bubalus bubalis",            # Trâu nước
    "Sus scrofa domesticus",      # Lợn nhà
    "Capra aegagrus hircus",      # Dê nhà
    "Ovis aries",                 # Cừu
    "Equus ferus caballus",       # Ngựa
    "Oryctolagus cuniculus",      # Thỏ nhà
    "Mesocricetus auratus",       # Chuột Hamster
    "Cyprinus carpio",            # Cá chép
    "Oreochromis niloticus",      # Cá rô phi
    "Columba livia domestica",    # Bồ câu
    "Passer domesticus",          # Chim sẻ
    "Rattus norvegicus",          # Chuột cống
    "Mus musculus",               # Chuột nhắt
    "Periplaneta americana",      # Gián Mỹ
    "Musca domestica",            # Ruồi nhà
    "Aedes aegypti",              # Muỗi vằn
    "Apis mellifera",             # Ong mật
    "Gekko gecko",                # Tắc kè
    "Hemidactylus frenatus",      # Thạch sùng
    "Canis latrans",              # Chó sói đồng cỏ (Common Coyote)
    "Procyon lotor",              # Gấu mèo (Raccoon)
    "Sciurus carolinensis",       # Sóc xám
    "Turdus migratorius",         # Chim hoét
    "Corvus brachyrhynchos",      # Quạ Mỹ
    "Pica pica",                  # Ác là
    "Sturnus vulgaris",           # Chim sáo đá
    "Carassius auratus",          # Cá vàng
    "Betta splendens",            # Cá xiêm
    "Poecilia reticulata",        # Cá bảy màu
    "Paracheirodon innesi",       # Cá Neon xanh
    "Pterophyllum scalare",       # Cá ông tiên
    "Xiphophorus hellerii",       # Cá đuôi kiếm
    "Trichogaster lalius",        # Cá sặc gấm
    "Corydoras paleatus",         # Cá chuột
    "Hypostomus plecostomus",     # Cá lau kính
    "Chromobotia macracanthus",   # Cá heo Mekong
    "Puntius tetrazona",          # Cá tứ vân
    "Danio rerio",                # Cá ngựa vằn (Zebrafish)
    "Amatitlania nigrofasciata",  # Cá Ali
    "Melopsittacus undulatus",    # Vẹt yến phụng
    "Nymphicus hollandicus",      # Vẹt mào (Cockatiel)
    "Serinus canaria",            # Chim yến
    "Taeniopygia guttata",        # Chim manh manh
    "Lonchura striata domestica", # Chim sắc Nhật
    "Anser anser domesticus"      # Ngỗng nhà
]

def create_dummy_image(folder_path, species_name, img_index, color):
    """Tạo ảnh giả lập bằng thư viện PIL"""
    width, height = 224, 224
    
    # Tạo biến thể màu sắc
    r, g, b = color
    variance = random.randint(-30, 30)
    final_color = (
        max(0, min(255, r + variance)),
        max(0, min(255, g + variance)),
        max(0, min(255, b + variance))
    )
    
    img = Image.new('RGB', (width, height), color=final_color)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
        
    # Format tên để hiển thị đẹp hơn trên ảnh
    display_name = species_name.replace(" ", "\n")
    text = f"{display_name}\nIdx: {img_index}"
    draw.text((10, 50), text, fill=(255, 255, 255), font=font)
    
    # Lưu file: dùng tên khoa học, thay khoảng trắng bằng gạch dưới
    file_name = f"{species_name.replace(' ', '_').lower()}_{img_index}.jpg"
    img.save(os.path.join(folder_path, file_name))

def main():
    # 1. Dọn dẹp và tạo thư mục gốc
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    os.makedirs(IMAGES_DIR)
    
    print(f"Đang khởi tạo dữ liệu tại: {BASE_DIR}...")
    
    csv_data = []
    
    # 2. Xử lý Loài Quý Hiếm (Label = 1)
    print("--- Đang tạo 50 loài quý hiếm (Scientific Names) ---")
    for species in rare_species_scientific:
        safe_folder_name = species.replace(" ", "_").lower()
        folder_path = os.path.join(IMAGES_DIR, safe_folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        for i in range(IMAGES_PER_SPECIES):
            # Màu đỏ sẫm cho loài quý hiếm
            create_dummy_image(folder_path, species, i, color=(139, 0, 0))
            
        csv_data.append({
            "species_name": species,
            "folder_name": safe_folder_name,
            "label": 1
        })

    # 3. Xử lý Loài Phổ Biến (Label = 0)
    print("--- Đang tạo 50 loài phổ biến (Scientific Names) ---")
    for species in common_species_scientific:
        safe_folder_name = species.replace(" ", "_").lower()
        folder_path = os.path.join(IMAGES_DIR, safe_folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        for i in range(IMAGES_PER_SPECIES):
            # Màu xanh lá sẫm cho loài phổ biến
            create_dummy_image(folder_path, species, i, color=(0, 100, 0))
            
        csv_data.append({
            "species_name": species,
            "folder_name": safe_folder_name,
            "label": 0
        })

    # 4. Xuất file CSV
    df = pd.DataFrame(csv_data)
    # Trộn ngẫu nhiên
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv(CSV_FILE, index=False)
    
    print("\n✅ HOÀN TẤT!")
    print(f"📁 Tổng số loài: {len(df)}")
    print(f"📄 File CSV: {CSV_FILE}")
    print(f"📂 Thư mục ảnh: {IMAGES_DIR}")
    print("\n5 dòng đầu của file CSV (Tên khoa học):")
    print(df.head())

if __name__ == "__main__":
    main()