import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app) # Quan trọng để React gọi được vào Python

# --- CẤU HÌNH ---
MODEL_PATH = 'animal_classifier.keras'
DATASET_DIR = 'animal_dataset/images'
LEGAL_DB_PATH = 'animal_dataset/legal_database.csv'

# 1. Đồng bộ nhãn (Labels) - Phải giống hệt lúc train.py chạy
class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
model = load_model(MODEL_PATH)
legal_df = pd.read_csv(LEGAL_DB_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    temp_path = "temp_predict.jpg"
    file.save(temp_path)

    try:
        # Preprocessing khớp hoàn toàn với train.py (MobileNetV2)
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = (img_array / 127.5) - 1.0 # Chuẩn hóa về [-1, 1]

        # Dự đoán
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        score = float(np.max(predictions[0]))

        # Lấy tên thư mục loài vật
        folder_name = class_names[predicted_index]
        
        # Truy vấn thông tin từ legal_database.csv
        animal_info = legal_df[legal_df['folder_name'] == folder_name]
        
        if animal_info.empty:
            return jsonify({
                "result": folder_name,
                "confidence": f"{score*100:.2f}",
                "error": "Nhận diện được nhưng không có dữ liệu pháp lý"
            })

        info = animal_info.iloc[0].to_dict()

        # TRẢ VỀ JSON KHỚP VỚI INTERFACE TRÊN LOOKUP.TSX
        return jsonify({
            "result": info['vietnamese_name'],
            "raw_name": folder_name,
            "confidence": f"{score*100:.2f}",
            "image_file": "/temp_predict.jpg", # Path để hiển thị lại ảnh
            "description": info['description'],
            "legal": {
                "vietnamese_name": info['vietnamese_name'],
                "legal_group": info['legal_group'],
                "decree": info['decree'],
                "penalty_warning": info['penalty_warning'],
                "farming_advice": info['farming_advice'],
                "status_code": info['status_code'], # 'success' | 'warning' | 'danger'
                "habitat": info.get('habitat', ''),
                "diet": info.get('diet', ''),
                "behavior": info.get('behavior', ''),
                "description": info['description']
            }
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)