import os
import shutil
# Cần cài đặt: pip install icrawler
from icrawler.builtin import BingImageCrawler

# --- CẤU HÌNH ---
BASE_DIR = "animal_dataset"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MAX_IMAGES = 100  # Số lượng ảnh muốn tải cho mỗi loài
THREADS = 4       # Số luồng tải song song (giảm nếu mạng yếu)

# --- DANH SÁCH LOÀI (Copy từ file generate_animals.py để đảm bảo đồng bộ) ---

rare_species_scientific = [
    "Pseudoryx nghetinhensis", "Rhinoceros sondaicus", "Panthera tigris corbetti", 
    "Rhinopithecus avunculus", "Panthera pardus orientalis", "Rafetus swinhoei", 
    "Elephas maximus", "Ailuropoda melanoleuca", "Manis pentadactyla", 
    "Pithecophaga jefferyi", "Phocoena sinus", "Gorilla beringei", 
    "Crocodylus siamensis", "Ciconia boyciana", "Panthera uncia", "Cuon alpinus", 
    "Nycticebus pygmaeus", "Ursus thibetanus", "Trachypithecus poliocephalus", 
    "Nomascus nasutus", "Eretmochelys imbricata", "Dermochelys coriacea", 
    "Pan troglodytes", "Pongo abelii", "Diceros bicornis", "Balaenoptera musculus", 
    "Equus ferus przewalskii", "Camelus ferus", "Grus japonensis", "Nipponia nippon", 
    "Panthera leo persica", "Acinonyx jubatus", "Lynx pardinus", "Addax nasomaculatus", 
    "Gazella leptoceros", "Oryx dammah", "Gorilla gorilla", "Pan paniscus", 
    "Indri indri", "Propithecus tattersalli", "Varecia variegata", 
    "Daubentonia madagascariensis", "Amblyrhynchus cristatus", "Chelonoidis nigra", 
    "Strigops habroptila", "Gymnogyps californianus", "Leucopsar rothschildi", 
    "Gavialis gangeticus", "Neofelis nebulosa", "Saiga tatarica"
]

common_species_scientific = [
    "Canis lupus familiaris", "Felis catus", "Gallus gallus domesticus", 
    "Anas platyrhynchos domesticus", "Bos taurus", "Bubalus bubalis", 
    "Sus scrofa domesticus", "Capra aegagrus hircus", "Ovis aries", 
    "Equus ferus caballus", "Oryctolagus cuniculus", "Mesocricetus auratus", 
    "Cyprinus carpio", "Oreochromis niloticus", "Columba livia domestica", 
    "Passer domesticus", "Rattus norvegicus", "Mus musculus", "Periplaneta americana", 
    "Musca domestica", "Aedes aegypti", "Apis mellifera", "Gekko gecko", 
    "Hemidactylus frenatus", "Canis latrans", "Procyon lotor", "Sciurus carolinensis", 
    "Turdus migratorius", "Corvus brachyrhynchos", "Pica pica", "Sturnus vulgaris", 
    "Carassius auratus", "Betta splendens", "Poecilia reticulata", "Paracheirodon innesi", 
    "Pterophyllum scalare", "Xiphophorus hellerii", "Trichogaster lalius", 
    "Corydoras paleatus", "Hypostomus plecostomus", "Chromobotia macracanthus", 
    "Puntius tetrazona", "Danio rerio", "Amatitlania nigrofasciata", 
    "Melopsittacus undulatus", "Nymphicus hollandicus", "Serinus canaria", 
    "Taeniopygia guttata", "Lonchura striata domestica", "Anser anser domesticus"
]

def download_images_for_species(species_list, label_name):
    print(f"\n--- BẮT ĐẦU TẢI ẢNH CHO NHÓM: {label_name.upper()} ---")
    
    total_species = len(species_list)
    
    for idx, species in enumerate(species_list):
        # Tạo tên thư mục chuẩn (giống file generate trước)
        safe_folder_name = species.replace(" ", "_").lower()
        save_dir = os.path.join(IMAGES_DIR, safe_folder_name)
        
        # Nếu thư mục chưa tồn tại thì tạo mới
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print(f"[{idx+1}/{total_species}] Đang tải ảnh cho: {species} -> {save_dir}")
        
        # Cấu hình bộ cào Bing
        # filters='photo': Chỉ tải ảnh chụp (tránh ảnh vẽ/clipart)
        crawler = BingImageCrawler(
            downloader_threads=THREADS,
            storage={'root_dir': save_dir},
            log_level='ERROR' # Chỉ hiện lỗi, ẩn bớt log thừa
        )
        
        # Bắt đầu cào
        # keyword: Tên khoa học thường cho kết quả chính xác hơn tên thường
        crawler.crawl(
            keyword=species, 
            filters=dict(type='photo'), 
            max_num=MAX_IMAGES,
            file_idx_offset='auto' # Tự động đánh số tiếp nếu thư mục đã có ảnh
        )

def main():
    print("--- CÔNG CỤ TẢI ẢNH ĐỘNG VẬT TỰ ĐỘNG ---")
    print(f"Lưu trữ tại: {IMAGES_DIR}")
    print(f"Số lượng mục tiêu: {MAX_IMAGES} ảnh/loài")
    
    # Kiểm tra xem thư mục gốc có tồn tại chưa (từ bước chạy generate_animals.py)
    if not os.path.exists(IMAGES_DIR):
        print(f"Cảnh báo: Thư mục {IMAGES_DIR} chưa tồn tại.")
        os.makedirs(IMAGES_DIR)

    # Tải ảnh cho loài quý hiếm
    download_images_for_species(rare_species_scientific, "Quý hiếm")
    
    # Tải ảnh cho loài phổ biến
    download_images_for_species(common_species_scientific, "Phổ biến")
    
    print("\n✅ HOÀN TẤT QUÁ TRÌNH TẢI ẢNH!")

if __name__ == "__main__":
    main()