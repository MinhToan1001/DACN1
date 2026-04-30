import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from flask import Flask, render_template, request, jsonify, send_from_directory
from torchvision.models import resnet50
from src.chatbot import ForestryChatbot

try:
    from src.legal.generate_legal_form import LegalFormGenerator, GENERATED_FORMS_DIR
except ImportError:
    from src.legal.generate_legal_form import LegalFormGenerator, GENERATED_FORMS_DIR

from src.knowledge import ExpertSystem

app = Flask(__name__)
GEMINI_API_KEY = "AIzaSyAQ0JxRzH8bsI2i8B8_QWU_Qn9bcz32tW4" 
chatbot = ForestryChatbot(api_key=GEMINI_API_KEY)
BASE_PATH = r"D:\HocTap\HK6\DACN1"
RULES_PATH = os.path.join(BASE_PATH, "rules")
MODELS_PATH = os.path.join(BASE_PATH, "models")
MODEL_PATH = os.path.join(MODELS_PATH, "best_animal_model.pt")
CLASS_NAMES_PATH = os.path.join(MODELS_PATH, "class_names.json")

form_gen = LegalFormGenerator()
expert_system = ExpertSystem(RULES_PATH)


def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(f"Không tìm thấy file {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


CLASS_NAMES = load_class_names()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model():
    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


model = load_trained_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


# ============================================================
# DEBUG: Xem cấu trúc JSON của một loài bất kỳ
# Gọi: GET /debug_species?name=Ophiophagus+hannah
# ============================================================
@app.route('/debug_species')
def debug_species():
    name = request.args.get("name", "")
    if not name:
        return jsonify({"error": "Provide ?name=scientific_name"})
    bio, legal = expert_system.kb.get_species_data(name)
    feats = expert_system.kb.get_identification_features(name)
    questions = expert_system.kb.get_adaptive_questions(name, 0.60)
    return jsonify({
        "bio_keys": list(bio.keys()),
        "bio_sample": {k: str(v)[:120] for k, v in bio.items()},
        "feats": feats,
        "questions_preview": questions,
        "legal_keys": list(legal.keys()),
    })


# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, idx = torch.max(probabilities, 0)

    predicted_species = CLASS_NAMES[idx.item()].replace('_', ' ')
    raw_conf = conf.item()

    print(f"[PREDICT] species={predicted_species}  raw_conf={raw_conf:.4f}")

    es_result = expert_system.initial_predict(predicted_species, raw_conf)
    es_result["raw_confidence"] = round(raw_conf, 6)
    return jsonify(es_result)


# ============================================================
@app.route('/answer_question', methods=['POST'])
def answer_question():
    """
    Body JSON:
    {
      "species": "Ophiophagus hannah",
      "current_confidence": 0.72,
      "answered": {"dac_diem_phan_biet": true, "mo_ta_ngoai_hinh": false}
    }
    """
    data = request.json
    species = data.get("species")
    current_conf = float(data.get("current_confidence", 0.5))
    answered = {k: bool(v) for k, v in data.get("answered", {}).items()}

    if not species:
        return jsonify({"error": "Missing species"}), 400

    print(f"[ANSWER] species={species}  cf={current_conf:.4f}  answered={answered}")
    result = expert_system.process_answer(species, current_conf, answered)
    return jsonify(result)


# ============================================================
@app.route('/generate_legal_doc', methods=['POST'])
def generate_legal_doc():
    data = request.json
    form_data = {
        "ten_loai_tieng_anh": data.get('species_name'),
        "ten_viet_nam": data.get('vietnamese_name'),
        "nhom_phap_ly": data.get('legal_group'),
        "ngay_thang_nam": "Ngày ... tháng ... năm 2026"
    }
    filename = f"Don_Ban_Giao_{data.get('species_name','species').replace(' ', '_')}.docx"
    file_path = form_gen.generate_form("mau_don_tu_nguyen.txt", form_data, filename)
    if file_path:
        return jsonify({"status": "success", "download_url": f"/download_form/{file_path}"})
    return jsonify({"status": "error", "message": "Failed to create file"}), 500


@app.route('/preview_legal_form', methods=['POST'])
def preview_legal_form():
    data = request.json
    form_data = {
        "ten_loai_tieng_anh": data.get('species_name'),
        "ten_viet_nam": data.get('vietnamese_name'),
        "nhom_phap_ly": data.get('legal_group'),
        "ngay_thang_nam": "Ngày ... tháng ... năm 2026"
    }
    content = form_gen.preview_form("mau_don_tu_nguyen.txt", form_data)
    if content:
        return jsonify({"status": "success", "content": content})
    return jsonify({"status": "error", "message": "Không tìm thấy mẫu đơn"}), 500


@app.route('/download_form/<filename>')
def download_form(filename):
    folder = str(GENERATED_FORMS_DIR)
    if os.path.exists(os.path.join(folder, filename)):
        return send_from_directory(directory=folder, path=filename, as_attachment=True)
    return "Không tìm thấy file", 404
@app.route('/chat', methods=['POST'])
def chat_api():
    """API Endpoint cho giao diện Chatbot"""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({"reply": "Dữ liệu không hợp lệ."}), 400
        
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({"reply": "Xin lỗi, tôi không nghe rõ câu hỏi."})

    # Gọi service xử lý
    reply_text = chatbot.get_response(user_message)
    
    return jsonify({"reply": reply_text})

if __name__ == '__main__':
    app.run(debug=True, port=5000)