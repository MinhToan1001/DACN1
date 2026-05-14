import requests
import os
import hashlib
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# =========================
# 1. LIST 104 LOÀI
# =========================

species_list = [
    # --- 130 LOÀI HOANG DÃ (NHÓM IB & IIB) ---
    # Lớp Thú
    "Panthera tigris", "Elephas maximus", "Ursus thibetanus", "Bos gaurus", "Manis javanica", 
    "Pygathrix nemaeus", "Nycticebus pygmaeus", "Capricornis milneedwardsii", "Muntiacus vuquangensis", 
    "Tragulus versicolor", "Lutra lutra", "Arctictis binturong", "Chrotogale owstoni", "Ratufa bicolor", 
    "Hystrix brachyura", "Petaurista philippensis", "Prionailurus bengalensis", "Cuon alpinus", 
    "Rusa unicolor", "Nomascus leucogenys", "Macaca arctoides", "Dugong dugon", "Neofelis nebulosa", 
    "Catopuma temminckii", "Pteropus vampyrus", "Helarctos malayanus", "Rhinopithecus avunculus", 
    "Prionodon pardicolor", "Neophocaena phocaenoides", "Cervus nippon",
    
    # Lớp Chim
    "Pavo muticus", "Buceros bicornis", "Grus antigone", "Platalea minor", "Pelecanus philippensis", 
    "Lophura diardi", "Rheinardia ocellata", "Garrulax yersini", "Gracula religiosa", "Psittacula alexandri", 
    "Spilornis cheela", "Pitta nympha", "Treron sieboldii", "Berenicornis comatus", "Leptoptilos javanicus", 
    "Otis bengalensis", "Asarcornis scutulata", "Tyto alba", "Ketupa zeylonensis", "Cissa chinensis", 
    "Harpactes erythrocephalus", "Megalaima lagrandieri", "Copsychus malabaricus", "Egretta eulophotes", 
    "Falco peregrinus", "Oriolus mellianus", "Eurylaimus javanicus", "Columba punicea", "Arborophila davidi", 
    "Polyplectron bicalcaratum", "Aquila nipalensis", "Ciconia episcopus", "Loriculus vernalis", 
    "Anthracoceros albirostris", "Garrulax milleti",
    
    # Lớp Bò sát
    "Crocodylus siamensis", "Varanus salvator", "Physignathus cocincinus", "Python bivittatus", 
    "Ophiophagus hannah", "Bungarus fasciatus", "Ptyas mucosa", "Gekko gecko", "Cuora galbinifrons", 
    "Pelochelys cantorii", "Chelonia mydas", "Platysternon megacephalum", "Eretmochelys imbricata", 
    "Dermochelys coriacea", "Shinisaurus crocodilurus", "Goniurosaurus catbaensis", "Coelognathus radiatus", 
    "Xenopeltis unicolor", "Eutropis multifasciata", "Acanthosaura", "Leiolepis belliana", "Ovophis monticola", 
    "Cylindrophis ruffus", "Indotestudo elongata", "Ahaetulla prasina", "Palea steindachneri", "Boiga cyanea", 
    "Enhydris enhydris", "Daboia siamensis", "Gekko badenii",
    
    # Lớp Lưỡng cư
    "Paramesotriton deloustali", "Tylototriton vietnamensis", "Theloderma corticale", 
    "Rhacophorus nigropalmatus", "Kaloula pulchra", "Leptobrachium", "Ichthyophis bannanicus", 
    "Brachytarsophrys feae", "Megophrys montana", "Amolops", "Bombina maxima", "Polypedates leucomystax", 
    "Odorrana", "Ingerophrynus galeatus", "Hylarana",
    
    # Lớp Côn trùng & Không xương sống
    "Teinopalpus aureus", "Troides helena", "Attacus atlas", "Cheirotonus macleayi", "Dorcus curvidens", 
    "Phyllium pulchrifolium", "Kallima inachus", "Eupatorus gracilicornis", "Lethocerus indicus", 
    "Tridacna gigas", "Pinctada margaritifera", "Cassis cornuta", "Charonia tritonis", "Tachypleus tridentatus", 
    "Panulirus ornatus", "Birgus latro", "Pharnacia", "Actias selene", "Papilio bianor", "Acropora",

    # --- 20 LOÀI PHỔ THÔNG / THÚ CƯNG (BACKGROUND CLASSES) ---
    "Canis familiaris",          # Chó nhà
    "Felis catus",               # Mèo nhà
    "Bos taurus",                # Bò nhà
    "Sus domesticus",            # Lợn nhà
    "Gallus gallus domesticus",  # Gà nhà
    "Anas platyrhynchos domesticus", # Vịt nhà
    "Equus caballus",            # Ngựa
    "Ovis aries",                # Cừu
    "Capra hircus",              # Dê
    "Oryctolagus cuniculus",     # Thỏ nhà
    "Cavia porcellus",           # Chuột lang (Guinea pig)
    "Columba livia domestica",   # Bồ câu nhà
    "Anser anser domesticus",    # Ngỗng nhà
    "Meleagris gallopavo",       # Gà tây
    "Bubalus bubalis",           # Trâu
    "Mesocricetus auratus",      # Chuột Hamster
    "Carassius auratus",         # Cá vàng
    "Cyprinus rubrofuscus",      # Cá chép Koi
    "Melopsittacus undulatus",   # Vẹt Yến Phụng (Budgerigar)
    "Serinus canaria"            # Chim Hoàng yến
]

print(f"Tổng số loài cần cào dữ liệu: {len(species_list)}")
# =========================
# 2. LEGAL INFO
# =========================

legal_info = {
    # --- LỚP THÚ (MAMMALIA) ---
    "Panthera_tigris": {"vn": "Hổ", "group": "IB", "status": "EN"},
    "Elephas_maximus": {"vn": "Voi châu Á", "group": "IB", "status": "EN"},
    "Ursus_thibetanus": {"vn": "Gấu ngựa", "group": "IB", "status": "VU"},
    "Bos_gaurus": {"vn": "Bò tót", "group": "IB", "status": "VU"},
    "Manis_javanica": {"vn": "Tê tê Java", "group": "IB", "status": "CR"},
    "Pygathrix_nemaeus": {"vn": "Chà vá chân nâu", "group": "IB", "status": "EN"},
    "Nycticebus_pygmaeus": {"vn": "Cu li nhỏ", "group": "IB", "status": "EN"},
    "Capricornis_milneedwardsii": {"vn": "Sơn dương", "group": "IB", "status": "NT"},
    "Muntiacus_vuquangensis": {"vn": "Mang lớn", "group": "IB", "status": "CR"},
    "Tragulus_versicolor": {"vn": "Cheo cheo lưng bạc", "group": "IB", "status": "CR"},
    "Lutra_lutra": {"vn": "Rái cá thường", "group": "IB", "status": "NT"},
    "Arctictis_binturong": {"vn": "Cầy mực", "group": "IB", "status": "VU"},
    "Chrotogale_owstoni": {"vn": "Cầy vằn bắc", "group": "IB", "status": "EN"},
    "Ratufa_bicolor": {"vn": "Sóc đen lớn", "group": "IIB", "status": "NT"},
    "Hystrix_brachyura": {"vn": "Nhím đuôi ngắn", "group": "IIB", "status": "LC"},
    "Petaurista_philippensis": {"vn": "Sóc bay lớn", "group": "IIB", "status": "LC"},
    "Prionailurus_bengalensis": {"vn": "Mèo rừng", "group": "IIB", "status": "LC"},
    "Cuon_alpinus": {"vn": "Sói đỏ", "group": "IB", "status": "EN"},
    "Rusa_unicolor": {"vn": "Nai", "group": "IIB", "status": "VU"},
    "Nomascus_leucogenys": {"vn": "Vượn đen má trắng", "group": "IB", "status": "CR"},
    "Macaca_arctoides": {"vn": "Khỉ mặt đỏ", "group": "IIB", "status": "VU"},
    "Dugong_dugon": {"vn": "Bò biển", "group": "IB", "status": "VU"},
    "Neofelis_nebulosa": {"vn": "Báo gấm", "group": "IB", "status": "VU"},
    "Catopuma_temminckii": {"vn": "Beo lửa", "group": "IB", "status": "NT"},
    "Pteropus_vampyrus": {"vn": "Dơi ngựa lớn", "group": "IIB", "status": "NT"},
    "Helarctos_malayanus": {"vn": "Gấu chó", "group": "IB", "status": "VU"},
    "Rhinopithecus_avunculus": {"vn": "Voọc mũi hếch", "group": "IB", "status": "CR"},
    "Prionodon_pardicolor": {"vn": "Cầy linsang đốm", "group": "IB", "status": "LC"},
    "Neophocaena_phocaenoides": {"vn": "Cá heo không vây", "group": "IB", "status": "VU"},
    "Cervus_nippon": {"vn": "Hươu sao", "group": "IIB", "status": "LC"},

    # --- LỚP CHIM (AVES) ---
    "Pavo_muticus": {"vn": "Công", "group": "IB", "status": "EN"},
    "Buceros_bicornis": {"vn": "Hồng hoàng", "group": "IB", "status": "VU"},
    "Grus_antigone": {"vn": "Sếu đầu đỏ", "group": "IB", "status": "VU"},
    "Platalea_minor": {"vn": "Cò thìa mặt đen", "group": "IB", "status": "EN"},
    "Pelecanus_philippensis": {"vn": "Bồ nông chân xám", "group": "IIB", "status": "NT"},
    "Lophura_diardi": {"vn": "Gà lôi hông tía", "group": "IIB", "status": "LC"},
    "Rheinardia_ocellata": {"vn": "Trĩ sao", "group": "IB", "status": "EN"},
    "Garrulax_yersini": {"vn": "Khướu ngực đốm", "group": "IB", "status": "EN"},
    "Gracula_religiosa": {"vn": "Yểng", "group": "IIB", "status": "LC"},
    "Psittacula_alexandri": {"vn": "Vẹt ngực đỏ", "group": "IIB", "status": "NT"},
    "Spilornis_cheela": {"vn": "Diều hâu hoa", "group": "IIB", "status": "LC"},
    "Pitta_nympha": {"vn": "Đuôi cụt bụng đỏ", "group": "IIB", "status": "VU"},
    "Treron_sieboldii": {"vn": "Cu xanh sáo", "group": "IIB", "status": "LC"},
    "Berenicornis_comatus": {"vn": "Niệng mào", "group": "IB", "status": "EN"},
    "Leptoptilos_javanicus": {"vn": "Già đẫy nhỏ", "group": "IB", "status": "VU"},
    "Otis_bengalensis": {"vn": "Ô tác Bengal", "group": "IB", "status": "CR"},
    "Asarcornis_scutulata": {"vn": "Ngan cánh trắng", "group": "IB", "status": "EN"},
    "Tyto_alba": {"vn": "Cú lợn lưng xám", "group": "IIB", "status": "LC"},
    "Ketupa_zeylonensis": {"vn": "Cú vọ mặt trắng", "group": "IIB", "status": "LC"},
    "Cissa_chinensis": {"vn": "Giẻ cùi", "group": "IIB", "status": "LC"},
    "Harpactes_erythrocephalus": {"vn": "Nuốc bụng đỏ", "group": "IIB", "status": "LC"},
    "Megalaima_lagrandieri": {"vn": "Cu rốc bụng nâu", "group": "IIB", "status": "LC"},
    "Copsychus_malabaricus": {"vn": "Chích chòe lửa", "group": "IIB", "status": "LC"},
    "Egretta_eulophotes": {"vn": "Cò trắng Trung Quốc", "group": "IB", "status": "VU"},
    "Falco_peregrinus": {"vn": "Cắt lớn", "group": "IIB", "status": "LC"},
    "Oriolus_mellianus": {"vn": "Tử anh", "group": "IB", "status": "EN"},
    "Eurylaimus_javanicus": {"vn": "Mỏ rộng xồm", "group": "IIB", "status": "LC"},
    "Columba_punicea": {"vn": "Bồ câu nâu", "group": "IIB", "status": "VU"},
    "Arborophila_davidi": {"vn": "Gà so cổ hung", "group": "IB", "status": "NT"},
    "Polyplectron_bicalcaratum": {"vn": "Gà tiền mặt vàng", "group": "IIB", "status": "LC"},
    "Aquila_nipalensis": {"vn": "Đại bàng thảo nguyên", "group": "IIB", "status": "EN"},
    "Ciconia_episcopus": {"vn": "Hạc cổ trắng", "group": "IIB", "status": "VU"},
    "Loriculus_vernalis": {"vn": "Vẹt lùn", "group": "IIB", "status": "LC"},
    "Anthracoceros_albirostris": {"vn": "Cao cát bụng trắng", "group": "IIB", "status": "LC"},
    "Garrulax_milleti": {"vn": "Khướu đầu đen", "group": "IB", "status": "NT"},

    # --- LỚP BÒ SÁT (REPTILIA) ---
    "Crocodylus_siamensis": {"vn": "Cá sấu nước ngọt", "group": "IB", "status": "CR"},
    "Varanus_salvator": {"vn": "Kỳ đà hoa", "group": "IIB", "status": "LC"},
    "Physignathus_cocincinus": {"vn": "Rồng đất", "group": "IIB", "status": "VU"},
    "Python_bivittatus": {"vn": "Trăn đất", "group": "IIB", "status": "VU"},
    "Ophiophagus_hannah": {"vn": "Rắn hổ chúa", "group": "IB", "status": "VU"},
    "Bungarus_fasciatus": {"vn": "Rắn cạp nong", "group": "IIB", "status": "LC"},
    "Ptyas_mucosa": {"vn": "Rắn ráo trâu", "group": "IIB", "status": "LC"},
    "Gekko_gecko": {"vn": "Tắc kè", "group": "IIB", "status": "LC"},
    "Cuora_galbinifrons": {"vn": "Rùa hộp trán vàng", "group": "IB", "status": "CR"},
    "Pelochelys_cantorii": {"vn": "Giải rùa", "group": "IB", "status": "CR"},
    "Chelonia_mydas": {"vn": "Vích", "group": "IB", "status": "EN"},
    "Platysternon_megacephalum": {"vn": "Rùa đầu to", "group": "IB", "status": "CR"},
    "Eretmochelys_imbricata": {"vn": "Đồi mồi", "group": "IB", "status": "CR"},
    "Dermochelys_coriacea": {"vn": "Rùa da", "group": "IB", "status": "VU"},
    "Shinisaurus_crocodilurus": {"vn": "Thằn lằn cá sấu", "group": "IB", "status": "EN"},
    "Goniurosaurus_catbaensis": {"vn": "Thạch sùng mí Cát Bà", "group": "IIB", "status": "EN"},
    "Coelognathus_radiatus": {"vn": "Rắn sọc dưa", "group": "IIB", "status": "LC"},
    "Xenopeltis_unicolor": {"vn": "Rắn mống", "group": "IIB", "status": "LC"},
    "Eutropis_multifasciata": {"vn": "Thằn lằn bóng", "group": "IIB", "status": "LC"},
    "Acanthosaura": {"vn": "Ô rô vảy", "group": "IIB", "status": "LC"},
    "Leiolepis_belliana": {"vn": "Nhông cát", "group": "IIB", "status": "LC"},
    "Ovophis_monticola": {"vn": "Rắn lục núi", "group": "IIB", "status": "LC"},
    "Cylindrophis_ruffus": {"vn": "Rắn trun", "group": "IIB", "status": "LC"},
    "Indotestudo_elongata": {"vn": "Rùa núi vàng", "group": "IIB", "status": "CR"},
    "Ahaetulla_prasina": {"vn": "Rắn lục cườm", "group": "IIB", "status": "LC"},
    "Palea_steindachneri": {"vn": "Ba ba gai", "group": "IIB", "status": "EN"},
    "Boiga_cyanea": {"vn": "Rắn rào xanh", "group": "IIB", "status": "LC"},
    "Enhydris_enhydris": {"vn": "Rắn bồng chì", "group": "IIB", "status": "LC"},
    "Daboia_siamensis": {"vn": "Nưa", "group": "IIB", "status": "LC"},
    "Gekko_badenii": {"vn": "Tắc kè núi Bà Đen", "group": "IIB", "status": "EN"},

    # --- LỚP LƯỠNG CƯ (AMPHIBIA) ---
    "Paramesotriton_deloustali": {"vn": "Cá cóc Tam Đảo", "group": "IB", "status": "VU"},
    "Tylototriton_vietnamensis": {"vn": "Cá cóc Việt Nam", "group": "IIB", "status": "EN"},
    "Theloderma_corticale": {"vn": "Ếch cây sần Bắc Bộ", "group": "IIB", "status": "LC"},
    "Rhacophorus_nigropalmatus": {"vn": "Ếch cây bay", "group": "IIB", "status": "LC"},
    "Kaloula_pulchra": {"vn": "Ưễnh ương", "group": "IIB", "status": "LC"},
    "Leptobrachium": {"vn": "Cóc mày", "group": "IIB", "status": "LC"},
    "Ichthyophis_bannanicus": {"vn": "Ếch giun", "group": "IIB", "status": "LC"},
    "Brachytarsophrys_feae": {"vn": "Cóc mắt kẹt", "group": "IIB", "status": "LC"},
    "Megophrys_montana": {"vn": "Cóc sừng", "group": "IIB", "status": "LC"},
    "Amolops": {"vn": "Ếch bám đá", "group": "IIB", "status": "LC"},
    "Bombina_maxima": {"vn": "Cóc tía bụng đỏ", "group": "IIB", "status": "LC"},
    "Polypedates_leucomystax": {"vn": "Ếch cây mép trắng", "group": "IIB", "status": "LC"},
    "Odorrana": {"vn": "Ếch suối", "group": "IIB", "status": "LC"},
    "Ingerophrynus_galeatus": {"vn": "Cóc tía", "group": "IIB", "status": "LC"},
    "Hylarana": {"vn": "Ếch xanh", "group": "IIB", "status": "LC"},

    # --- CÔN TRÙNG & KHÔNG XƯƠNG SỐNG ---
    "Teinopalpus_aureus": {"vn": "Bướm phượng đuôi kiếm", "group": "IB", "status": "DD"},
    "Troides_helena": {"vn": "Bướm phượng cánh chim", "group": "IIB", "status": "LC"},
    "Attacus_atlas": {"vn": "Bướm khế", "group": "IIB", "status": "NE"},
    "Cheirotonus_macleayi": {"vn": "Cua bay", "group": "IIB", "status": "NE"},
    "Dorcus_curvidens": {"vn": "Kẹp kìm răng lớn", "group": "IIB", "status": "NE"},
    "Phyllium_pulchrifolium": {"vn": "Bọ lá", "group": "IIB", "status": "NE"},
    "Kallima_inachus": {"vn": "Bướm lá khô", "group": "IIB", "status": "NE"},
    "Eupatorus_gracilicornis": {"vn": "Bọ hung năm sừng", "group": "IIB", "status": "NE"},
    "Lethocerus_indicus": {"vn": "Cà cuống", "group": "IIB", "status": "NE"},
    "Tridacna_gigas": {"vn": "Trai tai tượng khổng lồ", "group": "IB", "status": "VU"},
    "Pinctada_margaritifera": {"vn": "Trai ngọc môi đen", "group": "IIB", "status": "NE"},
    "Cassis_cornuta": {"vn": "Ốc kim khôi", "group": "IIB", "status": "NE"},
    "Charonia_tritonis": {"vn": "Ốc tù và", "group": "IB", "status": "NE"},
    "Tachypleus_tridentatus": {"vn": "Sam biển", "group": "IIB", "status": "EN"},
    "Panulirus_ornatus": {"vn": "Tôm hùm bông", "group": "IIB", "status": "LC"},
    "Birgus_latro": {"vn": "Cua xe tăng", "group": "IIB", "status": "VU"},
    "Pharnacia": {"vn": "Bọ que", "group": "IIB", "status": "NE"},
    "Actias_selene": {"vn": "Bướm mặt trăng", "group": "IIB", "status": "NE"},
    "Papilio_bianor": {"vn": "Bướm phượng xanh", "group": "IIB", "status": "NE"},
    "Acropora": {"vn": "San hô sừng hươu", "group": "IIB", "status": "NT"},

    # --- LOÀI PHỔ THÔNG / THÚ CƯNG (BACKGROUND CLASSES) ---
    "Canis_familiaris": {"vn": "Chó nhà", "group": "None", "status": "Domesticated"},
    "Felis_catus": {"vn": "Mèo nhà", "group": "None", "status": "Domesticated"},
    "Bos_taurus": {"vn": "Bò nhà", "group": "None", "status": "Domesticated"},
    "Sus_domesticus": {"vn": "Lợn nhà", "group": "None", "status": "Domesticated"},
    "Gallus_gallus_domesticus": {"vn": "Gà nhà", "group": "None", "status": "Domesticated"},
    "Anas_platyrhynchos_domesticus": {"vn": "Vịt nhà", "group": "None", "status": "Domesticated"},
    "Equus_caballus": {"vn": "Ngựa", "group": "None", "status": "Domesticated"},
    "Ovis_aries": {"vn": "Cừu", "group": "None", "status": "Domesticated"},
    "Capra_hircus": {"vn": "Dê", "group": "None", "status": "Domesticated"},
    "Oryctolagus_cuniculus": {"vn": "Thỏ nhà", "group": "None", "status": "Domesticated"},
    "Cavia_porcellus": {"vn": "Chuột lang", "group": "None", "status": "Domesticated"},
    "Columba_livia_domestica": {"vn": "Bồ câu nhà", "group": "None", "status": "Domesticated"},
    "Anser_anser_domesticus": {"vn": "Ngỗng nhà", "group": "None", "status": "Domesticated"},
    "Meleagris_gallopavo": {"vn": "Gà tây", "group": "None", "status": "Domesticated"},
    "Bubalus_bubalis": {"vn": "Trâu", "group": "None", "status": "Domesticated"},
    "Mesocricetus_auratus": {"vn": "Chuột Hamster", "group": "None", "status": "Domesticated"},
    "Carassius_auratus": {"vn": "Cá vàng", "group": "None", "status": "Domesticated"},
    "Cyprinus_rubrofuscus": {"vn": "Cá chép Koi", "group": "None", "status": "Domesticated"},
    "Melopsittacus_undulatus": {"vn": "Vẹt Yến Phụng", "group": "None", "status": "Domesticated"},
    "Serinus_canaria": {"vn": "Chim Hoàng yến", "group": "None", "status": "Domesticated"}
}

# =========================
# 3. TẠO THƯ MỤC ẢNH
# =========================

os.makedirs("images", exist_ok=True)

rows = []
def get_hash(url):
    return hashlib.md5(url.encode()).hexdigest()

def crawl_inat(species_name):
    print(f"  [iNat] Querying: {species_name}")
    url = "https://api.inaturalist.org/v1/observations"
    params = {"taxon_name": species_name, "per_page": 200, "photos": "true"}
    data = []
    try:
        r = requests.get(url, params=params, timeout=15).json()
        for obs in r.get("results", []):
            if not obs.get("photos"): continue
            lat = obs["geojson"]["coordinates"][1] if obs.get("geojson") else None
            lon = obs["geojson"]["coordinates"][0] if obs.get("geojson") else None
            for photo in obs["photos"]:
                img_url = photo["url"].replace("square", "large")
                data.append({
                    "image_id": get_hash(img_url),
                    "url": img_url,
                    "latitude": lat,
                    "longitude": lon,
                    "observed_time": obs.get("observed_on"),
                    "license": photo.get("license_code", ""),
                    "source": "inat"
                })
    except Exception as e:
        print(f"  [iNat] Lỗi: {e}")
    return data

def crawl_gbif(species_name):
    print(f"  [GBIF] Querying: {species_name}")
    url = "https://api.gbif.org/v1/occurrence/search"
    params = {"scientificName": species_name, "mediaType": "StillImage", "limit": 200}
    data = []
    try:
        r = requests.get(url, params=params, timeout=15).json()
        for obs in r.get("results", []):
            media = obs.get("media", [])
            lat = obs.get("decimalLatitude")
            lon = obs.get("decimalLongitude")
            event_date = obs.get("eventDate")
            for m in media:
                if m.get("type") == "StillImage":
                    img_url = m.get("identifier")
                    if not img_url: continue
                    data.append({
                        "image_id": get_hash(img_url),
                        "url": img_url,
                        "latitude": lat,
                        "longitude": lon,
                        "observed_time": event_date,
                        "license": m.get("license", ""),
                        "source": "gbif"
                    })
    except Exception as e:
        print(f"  [GBIF] Lỗi: {e}")
    return data

def download_task(item):
    try:
        if os.path.exists(item['file_path']): return
        img_data = requests.get(item['url'], timeout=10).content
        with open(item['file_path'], "wb") as f:
            f.write(img_data)
    except Exception as e:
        # In lỗi nhẹ để theo dõi nếu cần
        pass

# =========================
# MAIN RUNNER (ĐÃ FIX LỖI)
# =========================
os.makedirs("images", exist_ok=True)
all_rows = []

for sp in species_list:
    # sp đang là "Panthera tigris"
    sp_name = sp 
    sp_key = sp.replace(" ", "_") # Chuyển thành "Panthera_tigris" để tra cứu
    
    print(f"\n>>> Processing: {sp_name}")
    
    # Tối ưu: Dùng gạch dưới cho tên thư mục để tránh lỗi hệ điều hành sau này
    folder = f"images/{sp_key}" 
    os.makedirs(folder, exist_ok=True)
    
    # Gom dữ liệu từ 2 nguồn
    combined = crawl_inat(sp_name) + crawl_gbif(sp_name)
    
    # Lọc trùng Image ID 
    seen_ids = set()
    unique_combined = []
    for item in combined:
        if item['image_id'] not in seen_ids:
            seen_ids.add(item['image_id'])
            unique_combined.append(item)

    # FIX LỖI CRITICAL: Tra cứu bằng sp_key thay vì sp_name
    legal = legal_info.get(sp_key, {}) 
    sp_rows = []
    
    for item in unique_combined:
        img_path = f"{folder}/{item['image_id']}.jpg"
        row = {
            "image_id": item['image_id'],
            "scientific_name": sp_name,
            "common_name_vn": legal.get("vn", ""),     # Giờ sẽ lấy được data
            "legal_group": legal.get("group", ""),     # Giờ sẽ lấy được data
            "vnr_status": legal.get("status", ""),     # Giờ sẽ lấy được data
            "latitude": item['latitude'],
            "longitude": item['longitude'],
            "observed_time": item['observed_time'],
            "source": item['source'],
            "license": item['license'],
            "file_path": img_path,
            "url": item['url']
        }
        sp_rows.append(row)

    # Tải ảnh đa luồng
    print(f"  -> Downloading {len(sp_rows)} unique images...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_task, sp_rows)
        
    # FIX LOGIC TẢI ẢNH: Chỉ đưa vào CSV những ảnh đã thực sự được tải thành công
    successful_rows = [row for row in sp_rows if os.path.exists(row["file_path"])]
    all_rows.extend(successful_rows)
    print(f"  -> Thành công: {len(successful_rows)}/{len(sp_rows)} ảnh.")

# Lưu kết quả
df = pd.DataFrame(all_rows)
df.to_csv("dataset_combined.csv", index=False)
print("\n[SUCCESS] Done! Check 'dataset_combined.csv' and 'images/' folder.")