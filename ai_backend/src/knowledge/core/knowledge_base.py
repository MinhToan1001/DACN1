import os
import json
import re

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

    def _split_into_atomic_questions(self, trait_id, label, icon, text_content, base_priority, cf_yes, cf_no, cf_unknown=0.0):
        """Xé nhỏ đoạn văn bản và CHIA ĐỀU điểm tin cậy (CF) cho các ý nhỏ"""
        if not text_content or text_content == "Chưa có":
            return []

        parts = [p.strip() for p in re.split(r'[.;\n]+', str(text_content)) if len(p.strip()) > 5]
        num_parts = len(parts)
        
        if num_parts == 0:
            return []

        questions = []
        
        # ── CHIA ĐỀU ĐIỂM THƯỞNG/PHẠT ──
        # Ví dụ: Tổng phạt là -0.80, có 4 câu nhỏ -> mỗi câu chỉ phạt -0.20
        # Tránh việc rớt 1 ý nhỏ là rớt cả loài.
        cy_hi, cy_lo = cf_yes
        cn_hi, cn_lo = cf_no
        
        step_cy_hi, step_cy_lo = cy_hi / num_parts, cy_lo / num_parts
        step_cn_hi, step_cn_lo = cn_hi / num_parts, cn_lo / num_parts

        for i, part in enumerate(parts):
            short_detail = part if len(part) < 100 else part[:97] + "..."
            questions.append({
                "id": f"{trait_id}_{i}",
                "label": f"{label} ({i+1}/{num_parts})",
                "icon": icon,
                "detail": short_detail,
                "question": f"Mẫu vật có đặc điểm này không: '{part.lower()}'?",
                # Cộng i*0.01 để giữ đúng thứ tự câu trong cùng 1 nhóm mà không lấn sang nhóm khác
                "priority": base_priority + (i * 0.01), 
                "cf_yes_hi": step_cy_hi, "cf_no_hi": step_cn_hi,
                "cf_yes_lo": step_cy_lo, "cf_no_lo": step_cn_lo,
                "cf_unknown": cf_unknown
            })
        return questions

    def get_adaptive_questions(self, scientific_name: str, current_confidence: float):
        """Sinh danh sách câu hỏi theo chiến lược: Dễ trước, Chốt hạ sau"""
        bio_info, _ = self.get_species_data(scientific_name)
        if not bio_info:
            return []

        traits = bio_info.get("dac_diem_nhan_dang", {})
        valid_questions = []

        # ── CHIẾN LƯỢC ĐẶT CÂU HỎI (Priority tăng dần) ──
        
        # 1. MÔ TẢ NGOẠI HÌNH & KÍCH THƯỚC (Hỏi trước - Phạt rất nhẹ)
        valid_questions.extend(self._split_into_atomic_questions(
            trait_id="mo_ta_ngoai_hinh", label="Hình dáng & Màu sắc", icon="👁️",
            text_content=traits.get("mo_ta_ngoai_hinh"), base_priority=1.0, # Hỏi đầu tiên
            cf_yes=(0.20, 0.20), cf_no=(-0.15, -0.15) # Không có cũng chỉ bị trừ xíu
        ))

        # 2. MÔI TRƯỜNG SINH THÁI (Hỏi thứ 2 - Phạt nhẹ)
        valid_questions.extend(self._split_into_atomic_questions(
            trait_id="sinh_thai", label="Môi trường", icon="🌳",
            text_content=traits.get("sinh_thai"), base_priority=2.0,
            cf_yes=(0.15, 0.15), cf_no=(-0.10, -0.10)
        ))

        # 3. TẬP TÍNH (Hỏi thứ 3 - Phạt nhẹ)
        valid_questions.extend(self._split_into_atomic_questions(
            trait_id="tap_tinh", label="Tập tính", icon="🐾",
            text_content=traits.get("tap_tinh"), base_priority=3.0,
            cf_yes=(0.15, 0.15), cf_no=(-0.10, -0.10)
        ))

        # 4. ĐẶC ĐIỂM PHÂN BIỆT CHỐT HẠ (Hỏi cuối cùng - Phạt nặng nếu sai)
        # Vì đã chia đều (num_parts) nên dù phạt nặng (-0.8) nhưng mỗi câu sẽ gánh khoảng -0.2 đến -0.4
        valid_questions.extend(self._split_into_atomic_questions(
            trait_id="dac_diem_phan_biet", label="Đặc điểm then chốt", icon="🎯",
            text_content=traits.get("dac_diem_phan_biet"), base_priority=4.0, # Chốt hạ
            cf_yes=(0.40, 0.35), cf_no=(-0.80, -0.70) 
        ))

        # Sắp xếp toàn bộ câu hỏi theo độ ưu tiên từ thấp đến cao (Hỏi 1 -> 4)
        selected = sorted(valid_questions, key=lambda q: q["priority"])
        cf_key = "hi" if current_confidence >= 0.50 else "lo"

        return [{
            "id": q["id"],
            "label": q["label"],
            "icon": q["icon"],
            "detail": q["detail"],
            "question": q["question"],
            "cf_yes": q["cf_yes_hi"] if cf_key == "hi" else q["cf_yes_lo"],
            "cf_no":  q["cf_no_hi"] if cf_key == "hi" else q["cf_no_lo"],
            "cf_unknown": q.get("cf_unknown", 0.0),
            "answer_options": [
                {"value": "yes", "label": "Có", "cf": q["cf_yes_hi"] if cf_key == "hi" else q["cf_yes_lo"]},
                {"value": "no", "label": "Không", "cf": q["cf_no_hi"] if cf_key == "hi" else q["cf_no_lo"]},
                {"value": "unknown", "label": "Không biết", "cf": q.get("cf_unknown", 0.0)},
            ],
        } for q in selected]
    # Backward compat
    def get_all_questions(self, scientific_name: str,
                          current_confidence: float = 0.60) -> list:
        return self.get_adaptive_questions(scientific_name, current_confidence)
