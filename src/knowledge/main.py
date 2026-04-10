import json
import logging
from core.knowledge_base import KnowledgeBase
from engine.inference import InferenceEngine

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    # 1. Khởi tạo Knowledge Base và nạp dữ liệu từ 2 file JSON thật
    kb = KnowledgeBase()
    
    # LƯU Ý: Đảm bảo đường dẫn tới file JSON của bạn là chính xác
    # File chứa các luật IF-THEN (bạn tự định nghĩa dựa trên các features)
    kb.load_rules("data/rules.json") 
    
    # File chứa thông tin pháp lý bạn vừa tải lên (animal_rulebase.json)
    kb.load_legal_info("data/animal_rulebase.json")

    # 2. Khởi tạo Động cơ suy diễn
    engine = InferenceEngine(kb)

    # 3. Nạp output từ Deep Learning (nếu có)
    dl_predictions = {
        "Panthera tigris": 0.65, # Ví dụ DL model nhận diện ra Hổ với 65%
    }
    engine.load_dl_predictions(dl_predictions)

    # 4. Nạp các Facts từ user hoặc từ hệ thống trích xuất ảnh
    # (Ví dụ: user tích chọn các keywords hoặc OCR tự động quét được)
    user_input_facts = {
        "long_van_soc_den": 0.9,
        "co_to_khoe": 0.8
    }

    # 5. Chạy suy diễn
    result = engine.infer(user_facts=user_input_facts, debug=True)

    # 6. In kết quả
    print("\n" + "="*50)
    print("=== FINAL JSON OUTPUT ===")
    print("="*50)
    print(json.dumps(result, indent=4, ensure_ascii=False))

    print("\n" + "="*50)
    print("=== EXPLAINABLE AI LOG ===")
    print("="*50)
    print(engine.explain())

if __name__ == "__main__":
    main()