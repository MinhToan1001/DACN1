import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path

# Cấu hình đường dẫn
MODEL_PATH = "D:/hoctap/HK6/DACN1/models/best_animal_model.pt"
OUTPUT_DIR = Path("D:/hoctap/HK6/DACN1/config") # Nơi lưu ảnh Grad-CAM

# Định nghĩa các lớp (cần khớp với JSON của bạn)
CLASS_NAMES = [
    "Berenicornis comatus,"
    "Rusa unicolor,"
    "Lophura diardi,"
    "Pitta nympha,"
    "Actias selene,"
    "Egretta eulophotes,"
    "Prionailurus bengalensis,"
    "Indotestudo elongata,"
    "Panthera tigris,"
    "Pavo muticus,"
    "Anas platyrhynchos domesticus,"
    "Polyplectron bicalcaratum,"
    "Platalea minor,"
    "Python bivittatus,"
    "Helarctos malayanus,"
    "Platysternon megacephalum,"
    "Elephas maximus,"
    "Ratufa bicolor,"
    "Pygathrix nemaeus,"
    "Macaca arctoides,"
    "Anthracoceros albirostris,"
    "Loriculus vernalis,"
    "Lethocerus indicus,"
    "Troides helena,"
    "Felis catus,"
    "Varanus salvator,"
    "Charonia tritonis,"
    "Copsychus malabaricus,"
    "Physignathus cocincinus,"
    "Pelecanus philippensis,"
    "Gekko gecko,"
    "Spilornis cheela,"
    "Melopsittacus undulatus,"
    "Ketupa zeylonensis,"
    "Grus antigone,"
    "Birgus latro,"
    "Cuon alpinus,"
    "Pteropus vampyrus,"
    "Psittacula alexandri,"
    "Leptoptilos javanicus,"
    "Manis javanica,"
    "Nomascus leucogenys,"
    "Chelonia mydas,"
    "Buceros bicornis,"
    "Cervus nippon,"
    "Canis familiaris,"
    "Ophiophagus hannah,"
    "Tyto alba,"
    "Panulirus ornatus,"
    "Dorcus curvidens,"
    "Harpactes erythrocephalus,"
    "Arctictis binturong,"
    "Gallus gallus domesticus,"
    "Ahaetulla prasina,"
    "Ciconia episcopus,"
    "Eretmochelys imbricata,"
    "Petaurista philippensis,"
    "Ptyas mucosa,"
    "Oryctolagus cuniculus,"
    "Neofelis nebulosa,"
    "Ursus thibetanus,"
    "Aquila nipalensis,"
    "Sus domesticus,"
    "Bos gaurus,"
    "Eutropis multifasciata,"
    "Meleagris gallopavo,"
    "Bos taurus,"
    "Dermochelys coriacea,"
    "Crocodylus siamensis,"
    "Dugong dugon,"
    "Ovis aries,"
    "Kaloula pulchra,"
    "Equus caballus,"
    "Amolops,"
    "Tachypleus tridentatus,"
    "Falco peregrinus,"
    "Bubalus bubalis,"
    "Lutra lutra,"
    "Hylarana,"
    "Eurylaimus javanicus,"
    "Shinisaurus crocodilurus,"
    "Capra hircus,"
    "Gracula religiosa,"
    "Hystrix brachyura,"
    "Theloderma corticale,"
    "Columba punicea,"
    "Otis bengalensis,"
    "Palea steindachneri,"
    "Cuora galbinifrons,"
    "Gekko badenii,"
    "Capricornis milneedwardsii,"
    "Nycticebus pygmaeus,"
    "Teinopalpus aureus,"
    "Garrulax yersini,"
    "Catopuma temminckii,"
    "Prionodon pardicolor,"
    "Garrulax milleti,"
    "Pelochelys cantorii,"
    "Paramesotriton deloustali,"
    "Rheinardia ocellata,"
    "Chrotogale owstoni,"
    "Rhinopithecus avunculus,"
    "Arborophila davidi"
] 

def get_prediction_for_RL_ES(image_path, model, device):
    """
    Hàm này xuất ra độ tự tin (Confidence Score) cho RL và ES.
    Output ví dụ: {"Panthera tigris": 0.85, "Elephas maximus": 0.10}
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

    # Đóng gói thành Dictionary để truyền qua Backend / Hệ chuyên gia
    result_dict = {CLASS_NAMES[i]: float(probs) for i, probs in enumerate(probabilities)}
    
    # Sắp xếp từ cao xuống thấp
    sorted_result = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_result

def generate_gradcam_heatmap(image_path, model, device):
    """
    Tạo bản đồ nhiệt Explainable AI (Grad-CAM) lưu để báo cáo đồ án.
    """
    target_layers = [model.layer4[-1]] # Lớp tích chập cuối cùng của ResNet50
    cam = GradCAM(model=model, target_layers=target_layers)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Chạy Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # Chuẩn bị ảnh gốc để đè Heatmap lên
    img_resized = img.resize((224, 224))
    rgb_img = np.float32(img_resized) / 255
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Lưu ảnh giải thích ra file
    output_path = OUTPUT_DIR / f"gradcam_result.png"
    cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    print(f"🔥 Đã xuất bản đồ nhiệt Grad-CAM tại: {output_path}")
    return output_path