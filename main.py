import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import torch.nn.functional as F

# Import hàm build model từ source code của bạn (dựa theo app.py bạn cung cấp)
from src.models.model_mobilenet import build_mobilenetv2_model

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN & THÔNG SỐ
# ==========================================
# Cách 1: Dùng Raw String (thêm chữ r)
MODEL_DIR = r"D:\HocTap\HK6\DACN1\models"
MODEL_FILENAME = "best_animal_model.pt" 
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

RULEBASE_PATH = "D:/HocTap/HK6/DACN1/rules/animal_rulebase.json"
BIOLOGICAL_PATH = "D:/HocTap/HK6/DACN1/rules/species_biological_features.json"

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Phân Loại Động Vật Quý Hiếm", layout="wide")

# ==========================================
# CÁC HÀM TẢI DỮ LIỆU & MODEL (Dùng Cache để tối ưu)
# ==========================================
@st.cache_resource
@st.cache_resource
def load_model_and_classes():
    # 1. Khai báo trực tiếp tên các folder chứa ảnh lúc bạn train
    # ⚠️ LƯU Ý QUAN TRỌNG: Bạn PHẢI sắp xếp tên các loài theo đúng thứ tự Alphabet (A-Z) 
    # vì mặc định PyTorch DataLoader (ImageFolder) sẽ đọc folder theo thứ tự chữ cái.
    class_names = [
        'actias_selene',
        'ahaetulla_prasina',
        'amolops',
        'anas_platyrhynchos_domesticus',
        'anthracoceros_albirostris',
        'aquila_nipalensis',
        'arborophila_davidi',
        'arctictis_binturong',
        'berenicornis_comatus',
        'birgus_latro',
        'bos_gaurus',
        'bos_taurus',
        'bubalus_bubalis',
        'buceros_bicornis',
        'canis_familiaris',
        'capra_hircus',
        'capricornis_milneedwardsii',
        'catopuma_temminckii',
        'cervus_nippon',
        'charonia_tritonis',
        'chelonia_mydas',
        'chrotogale_owstoni',
        'ciconia_episcopus',
        'columba_punicea',
        'copsychus_malabaricus',
        'crocodylus_siamensis',
        'cuon_alpinus',
        'cuora_galbinifrons',
        'dermochelys_coriacea',
        'dorcus_curvidens',
        'dugong_dugon',
        'egretta_eulophotes',
        'elephas_maximus',
        'equus_caballus',
        'eretmochelys_imbricata',
        'eurylaimus_javanicus',
        'eutropis_multifasciata',
        'falco_peregrinus',
        'felis_catus',
        'gallus_gallus_domesticus',
        'garrulax_milleti',
        'garrulax_yersini',
        'gekko_badenii',
        'gekko_gecko',
        'gracula_religiosa',
        'grus_antigone',
        'harpactes_erythrocephalus',
        'helarctos_malayanus',
        'hylarana',
        'hystrix_brachyura',
        'indotestudo_elongata',
        'kaloula_pulchra',
        'ketupa_zeylonensis',
        'leptoptilos_javanicus',
        'lethocerus_indicus',
        'lophura_diardi',
        'loriculus_vernalis',
        'lutra_lutra',
        'macaca_arctoides',
        'manis_javanica',
        'meleagris_gallopavo',
        'melopsittacus_undulatus',
        'neofelis_nebulosa',
        'nomascus_leucogenys',
        'nycticebus_pygmaeus',
        'ophiophagus_hannah',
        'oryctolagus_cuniculus',
        'otis_bengalensis',
        'ovis_aries',
        'palea_steindachneri',
        'panthera_tigris',
        'panulirus_ornatus',
        'paramesotriton_deloustali',
        'pavo_muticus',
        'pelecanus_philippensis',
        'pelochelys_cantorii',
        'petaurista_philippensis',
        'physignathus_cocincinus',
        'pitta_nympha',
        'platalea_minor',
        'platysternon_megacephalum',
        'polyplectron_bicalcaratum',
        'prionailurus_bengalensis',
        'prionodon_pardicolor',
        'psittacula_alexandri',
        'pteropus_vampyrus',
        'ptyas_mucosa',
        'pygathrix_nemaeus',
        'python_bivittatus',
        'ratufa_bicolor',
        'rheinardia_ocellata',
        'rhinopithecus_avunculus',
        'rusa_unicolor',
        'shinisaurus_crocodilurus',
        'spilornis_cheela',
        'sus_domesticus',
        'tachypleus_tridentatus',
        'teinopalpus_aureus',
        'theloderma_corticale',
        'troides_helena',
        'tyto_alba',
        'ursus_thibetanus',
        'varanus_salvator',
    ]
    
    num_classes = len(class_names)
    
    # 2. Khởi tạo model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mobilenetv2_model(num_classes=num_classes)
    
    # 3. Load weights từ file .pth
# 3. Load weights từ file .pth
    
    # --- ĐOẠN CODE DEBUG THÊM VÀO ---
    st.warning(f"Đang tìm model tại chính xác đường dẫn này: {MODEL_PATH}")
    if os.path.exists(MODEL_DIR):
        files_in_dir = os.listdir(MODEL_DIR)
        st.info(f"Các file thực tế đang có trong thư mục models gồm: {files_in_dir}")
    else:
        st.error(f"Thư mục {MODEL_DIR} hoàn toàn không tồn tại!")
    # ---------------------------------

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        st.success("✅ Đã load model thành công!")
    else:
        st.error(f"❌ Vẫn không tìm thấy model tại {MODEL_PATH}")
    model.to(device)
    model.eval()
    
    # QUAN TRỌNG NHẤT LÀ DÒNG NÀY (Trả về kết quả):
    return model, class_names, device

@st.cache_data
def load_json_data():
    # Load Rulebase
    with open(RULEBASE_PATH, 'r', encoding='utf-8') as f:
        rulebase = json.load(f)
        
    # Load Biological Features
    with open(BIOLOGICAL_PATH, 'r', encoding='utf-8') as f:
        bio_list = json.load(f)
        # Chuyển list thành dict với key là scientific_name để tra cứu cho nhanh O(1)
        biological = {item["scientific_name"]: item for item in bio_list}
        
    return rulebase, biological

# ==========================================
# HÀM DỰ ĐOÁN
# ==========================================
def predict_image(image, model, class_names, device):
    # Pipeline biến đổi ảnh (giống với lúc train)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
    sci_name = class_names[predicted_idx.item()]
    conf_score = confidence.item() * 100
    
    return sci_name, conf_score

# ==========================================
# XÂY DỰNG GIAO DIỆN CHÍNH
# ==========================================
def main():
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Menu", ["Prediction", "Analytics", "Settings"])
    
    # Load data & model
    model, class_names, device = load_model_and_classes()
    rulebase_data, bio_data = load_json_data()

    if page == "Prediction":
        st.title("Phân Loại Động Vật Quý Hiếm")
        
        # Upload ảnh
        uploaded_file = st.file_uploader("Kéo thả hoặc tải ảnh lên tại đây (Drag and drop file here)", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            # Layout 2 cột như trong hình
            col1, col2 = st.columns([1, 1])
            
            image = Image.open(uploaded_file).convert("RGB")
            
            with col1:
                st.markdown("### Preview")
                st.image(image, use_column_width=True)
                
            # Chạy dự đoán
            sci_name, conf_score = predict_image(image, model, class_names, device)
            
            # Lấy thông tin từ JSON
            animal_rules = rulebase_data.get(sci_name, {})
            animal_bio = bio_data.get(sci_name, {})
            common_name = animal_rules.get("common_name", animal_bio.get("ten_viet_nam", "Đang cập nhật..."))

            with col2:
                st.markdown("### Kết quả chẩn đoán:")
                # Thẻ kết quả
                st.success(f"""
                **Tên khoa học:** {sci_name}  
                **Tên thông thường:** {common_name}  
                **Độ tin cậy:** {conf_score:.2f}%
                """)
            
            st.markdown("---")
            
            # Tabs hiển thị chi tiết (như trong hình mẫu)
            tab1, tab2 = st.tabs(["🧬 Đặc điểm sinh học", "⚖️ Quy định pháp luật"])
            
            with tab1:
                if animal_bio:
                    st.markdown(f"**Tên Việt Nam:** {animal_bio.get('ten_viet_nam', '')}")
                    dac_diem = animal_bio.get("dac_diem_nhan_dang", {})
                    st.markdown(f"**Mô tả ngoại hình:** {dac_diem.get('mo_ta_ngoai_hinh', 'N/A')}")
                    st.markdown(f"**Thức ăn:** {dac_diem.get('thuc_an', 'N/A')}")
                    st.markdown(f"**Tập tính:** {dac_diem.get('tap_tinh', 'N/A')}")
                    st.markdown(f"**Sinh thái:** {dac_diem.get('sinh_thai', 'N/A')}")
                else:
                    st.warning("Không tìm thấy thông tin sinh học cho loài này trong cơ sở dữ liệu.")
                    
            with tab2:
                if animal_rules:
                    st.markdown(f"**Nhóm pháp lý:** {animal_rules.get('legal_group', 'N/A')}")
                    legal_adv = animal_rules.get("legal_advice", {})
                    st.markdown(f"**Tên nhóm:** {legal_adv.get('group_name', 'N/A')}")
                    
                    st.markdown("**Các khung hình phạt hình sự:**")
                    penalties = legal_adv.get("criminal_penalties", {})
                    for khung, details in penalties.items():
                        st.markdown(f"- **{khung.replace('_', ' ')}**")
                        if isinstance(details, list):
                            for d in details:
                                st.markdown(f"  + {d}")
                        else:
                            st.markdown(f"  + {details}")
                else:
                    st.warning("Không tìm thấy quy định pháp luật cho loài này trong cơ sở dữ liệu.")

if __name__ == "__main__":
    main()