import os
import json


class KnowledgeBase:
    """Lưu trữ và truy xuất kiến thức từ rules JSON"""

    def __init__(self, rules_path: str):
        self.rules_path = rules_path
        self.bio_data = self._load_json("species_biological_features.json")
        self.legal_data = self._load_json("animal_rulebase.json")

    def _load_json(self, filename: str):
        path = os.path.join(self.rules_path, filename)
        if not os.path.exists(path):
            print(f"[WARNING] Không tìm thấy file: {path}")
            return [] if filename == "species_biological_features.json" else {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ------------------------------------------------------------------
    def get_species_data(self, scientific_name: str):
        """Lấy thông tin sinh học + pháp lý, tìm kiếm linh hoạt"""
        bio_data = {}
        
        # Chuẩn hóa tên đầu vào: viết thường và đổi khoảng trắng thành dấu gạch dưới
        search_name = scientific_name.lower().replace(" ", "_")

        for item in (self.bio_data if isinstance(self.bio_data, list) else []):
            sn = item.get("scientific_name", "")
            # Chuẩn hóa tên trong database
            sn_normalized = sn.lower().replace(" ", "_")
            
            if sn_normalized == search_name:
                bio_data = item
                break

        legal_data = {}
        if isinstance(self.legal_data, dict):
            for k, v in self.legal_data.items():
                k_normalized = k.lower().replace(" ", "_")
                if k_normalized == search_name:
                    legal_data = v
                    break

        return bio_data, legal_data
    # ------------------------------------------------------------------
    @staticmethod
    def _get_any(d: dict, *keys) -> str:
        """Thử lần lượt nhiều key tên, trả về chuỗi đầu tiên có giá trị"""
        for key in keys:
            val = d.get(key, "")
            if val and str(val).strip() and "Chưa có" not in str(val):
                return str(val).strip()
        return ""

    # ------------------------------------------------------------------
    def get_identification_features(self, scientific_name: str) -> dict:
        """
        Chuẩn hoá đặc điểm nhận dạng.
        Hỗ trợ cả JSON lồng nhau (dac_diem_nhan_dang) lẫn flat.
        """
        bio_data, _ = self.get_species_data(scientific_name)

        nested = bio_data.get("dac_diem_nhan_dang", {})
        if not isinstance(nested, dict) or not nested:
            nested = bio_data  # fallback sang flat

        result = {
            "mo_ta_ngoai_hinh": self._get_any(nested,
                "mo_ta_ngoai_hinh", "ngoai_hinh", "mo_ta", "appearance", "morphology"),
            "thuc_an": self._get_any(nested,
                "thuc_an", "diet", "food"),
            "tap_tinh": self._get_any(nested,
                "tap_tinh", "behavior", "behaviour"),
            "sinh_thai": self._get_any(nested,
                "sinh_thai", "habitat", "ecology", "moi_truong"),
            "phan_bo_viet_nam": self._get_any(nested,
                "phan_bo_viet_nam", "phan_bo", "distribution", "range"),
            "dac_diem_phan_biet": self._get_any(nested,
                "dac_diem_phan_biet", "distinguishing_features",
                "key_features", "nhan_dang", "dac_trung"),
        }

        print(f"[KB] {scientific_name} → "
              f"ngoai_hinh={bool(result['mo_ta_ngoai_hinh'])}, "
              f"phan_biet={bool(result['dac_diem_phan_biet'])}, "
              f"phan_bo={bool(result['phan_bo_viet_nam'])}, "
              f"tap_tinh={bool(result['tap_tinh'])}")
        return result

    # ------------------------------------------------------------------
    def get_adaptive_questions(self, scientific_name: str, current_confidence: float):
        """Sinh danh sách câu hỏi xác nhận (dựa trên confidence)"""
        bio_info, _ = self.get_species_data(scientific_name)
        if not bio_info:
            return []

        traits = bio_info.get("dac_diem_nhan_dang", {})
        valid_questions = []

        # 1. ƯU TIÊN SỐ 1: ĐẶC ĐIỂM PHÂN BIỆT (Knock-out Question)
        if traits.get("dac_diem_phan_biet"):
            valid_questions.append({
                "id": "dac_diem_phan_biet",
                "label": "Đặc điểm nhận dạng then chốt",
                "icon": "🎯",
                "detail": traits.get("dac_diem_phan_biet"),
                "question": "Con vật này CÓ CHÍNH XÁC đặc điểm độc nhất sau đây không?",
                "priority": 1,         # LUÔN LUÔN HỎI ĐẦU TIÊN
                "cf_yes_hi": 0.40,     "cf_no_hi": -0.80,  # Sai một phát là trừ 80% niềm tin -> Đánh rớt luôn
                "cf_yes_lo": 0.35,     "cf_no_lo": -0.70   
            })

        # 2. ƯU TIÊN SỐ 2: NGOẠI HÌNH & KÍCH THƯỚC (Hỏi để củng cố thêm)
        if traits.get("mo_ta_ngoai_hinh"):
            valid_questions.append({
                "id": "mo_ta_ngoai_hinh",
                "label": "Hình dáng & Màu sắc",
                "icon": "👁️",
                "detail": traits.get("mo_ta_ngoai_hinh"),
                "question": "Về tổng thể ngoại hình (màu sắc, hình dáng), nó có giống mô tả này không?",
                "priority": 2,
                "cf_yes_hi": 0.25,     "cf_no_hi": -0.30,  # Trừ vừa phải nếu nhìn không rõ
                "cf_yes_lo": 0.20,     "cf_no_lo": -0.20
            })

        # 3. ƯU TIÊN SỐ 3: MÔI TRƯỜNG TÌM THẤY & TẬP TÍNH
        if traits.get("sinh_thai"):
            valid_questions.append({
                "id": "sinh_thai",
                "label": "Môi trường phát hiện",
                "icon": "🌳",
                "detail": traits.get("sinh_thai"),
                "question": "Bạn có bắt gặp mẫu vật ở khu vực/môi trường như thế này không?",
                "priority": 3,
                "cf_yes_hi": 0.15,     "cf_no_hi": -0.10,  # Sinh thái sai vẫn có thể do con vật đi lạc, trừ rất ít
                "cf_yes_lo": 0.15,     "cf_no_lo": -0.10
            })

        if traits.get("tap_tinh"):
            valid_questions.append({
                "id": "tap_tinh",
                "label": "Tập tính",
                "icon": "🐾",
                "detail": traits.get("tap_tinh"),
                "question": "Nó có biểu hiện tập tính như trên không?",
                "priority": 4,
                "cf_yes_hi": 0.10,     "cf_no_hi": -0.05,
                "cf_yes_lo": 0.10,     "cf_no_lo": -0.05
            })

        if not valid_questions:
            return []

        # -- XỬ LÝ LỌC SỐ LƯỢNG CÂU HỎI THEO CONFIDENCE --
        if current_confidence >= 0.70:
            selected = sorted(valid_questions, key=lambda q: q["priority"])[:2]
            cf_key = "hi"
        elif current_confidence >= 0.50:
            selected = sorted(valid_questions, key=lambda q: q["priority"])[:3]
            cf_key = "hi"
        else:
            selected = sorted(valid_questions, key=lambda q: q["priority"])
            cf_key = "lo"

        # Đóng gói trả về UI
        return [{
            "id": q["id"],
            "label": q["label"],
            "icon": q["icon"],
            "detail": q["detail"],
            "question": q["question"],
            "cf_yes": q[f"cf_yes_{cf_key}"],
            "cf_no":  q[f"cf_no_{cf_key}"]
        } for q in selected]
    # Backward compat
    def get_all_questions(self, scientific_name: str,
                          current_confidence: float = 0.60) -> list:
        return self.get_adaptive_questions(scientific_name, current_confidence)