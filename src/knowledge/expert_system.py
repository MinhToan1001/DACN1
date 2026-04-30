from .core.knowledge_base import KnowledgeBase
from .core.fuzzy_logic import FuzzyLogic, get_fuzzy_assessment
from .engine.inference import InferenceEngine


class ExpertSystem:
    """Hệ chuyên gia: fuzzy + forward + backward inference + interactive Q&A"""

    def __init__(self, rules_path: str):
        self.kb = KnowledgeBase(rules_path)
        self.inference = InferenceEngine(self.kb)
        self.fuzzy = FuzzyLogic()

    # ------------------------------------------------------------------
    def _build_base(self, bio_info: dict, predicted_species: str,
                    confidence_raw: float) -> dict:
        """Tạo base dict chung với tên tiếng Việt đúng"""
        # Thử nhiều key có thể chứa tên tiếng Việt
        vn_name = (
            bio_info.get("ten_viet_nam") or
            bio_info.get("vietnamese_name") or
            bio_info.get("ten_loai") or
            bio_info.get("common_name") or
            "Chưa có tên tiếng Việt"
        )
        fuzzy_result = get_fuzzy_assessment(confidence_raw)
        return {
            "species": predicted_species,
            "vietnamese_name": vn_name,
            "confidence": round(confidence_raw * 100, 2),
            "fuzzy_status": fuzzy_result["fuzzy_status"],
            "fuzzy_message": fuzzy_result["message"],
        }

    # ------------------------------------------------------------------
    def initial_predict(self, predicted_species: str, raw_confidence: float) -> dict:
        """
        Xử lý lần đầu khi model AI trả về kết quả.
          ≥ 90%  → SUCCESS: hiển thị đầy đủ sinh học + pháp lý
          31-89% → ASKING:  gửi danh sách câu hỏi thích nghi
          ≤ 30%  → REJECTED
        """
        bio_info, legal_info = self.kb.get_species_data(predicted_species)
        base = self._build_base(bio_info, predicted_species, raw_confidence)

        # ── TRƯỜNG HỢP 1: TỰ TIN CAO ──────────────────────────────────
        if raw_confidence >= 0.90:
            forward = self.inference.forward_chaining({"predicted_species": predicted_species})
            # Đảm bảo biology chứa đặc điểm nhận dạng đã chuẩn hoá
            bio_full = forward.get("biology") or bio_info
            bio_feats = self.kb.get_identification_features(predicted_species)
            # Merge vào dac_diem_nhan_dang để frontend đọc đúng
            if isinstance(bio_full, dict):
                if "dac_diem_nhan_dang" not in bio_full or not bio_full["dac_diem_nhan_dang"]:
                    bio_full["dac_diem_nhan_dang"] = bio_feats
                else:
                    # Bổ sung các field còn thiếu
                    for k, v in bio_feats.items():
                        if v and not bio_full["dac_diem_nhan_dang"].get(k):
                            bio_full["dac_diem_nhan_dang"][k] = v

            return {
                **base,
                "status": "SUCCESS",
                "message": "Hệ thống tin tưởng cao vào dự đoán này.",
                "biology": bio_full,
                "legal": forward.get("legal"),
                "inferred_legal_group": forward.get("legal_group"),
            }

        # ── TRƯỜNG HỢP 2: QUÁ THẤP ────────────────────────────────────
        if raw_confidence <= 0.30:
            return {
                **base,
                "status": "REJECTED",
                "message": (f"Độ tin cậy quá thấp ({round(raw_confidence*100, 2)}%). "
                            "Loài này chưa có trong dữ liệu hệ thống hoặc không thể nhận dạng."),
            }

        # ── TRƯỜNG HỢP 3: CẦN HỎI THÊM (câu hỏi thích nghi) ──────────
        questions = self.kb.get_adaptive_questions(predicted_species, raw_confidence)
        return {
            **base,
            "status": "ASKING",
            "message": (f"Độ tin cậy {round(raw_confidence*100, 2)}% chưa đạt ngưỡng (90%). "
                        "Vui lòng xác nhận một số đặc điểm sinh học bên dưới."),
            "questions": questions,
        }

    # ------------------------------------------------------------------
    def process_answer(self, predicted_species: str,
                       current_confidence: float, answered: dict) -> dict:
        """
        Nhận toàn bộ câu đã trả lời, tính CF mới, quyết định bước tiếp.
        answered: {question_id: True/False, ...}
        current_confidence: CF hiện tại (0.0 – 1.0)
        """
        bio_info, legal_info = self.kb.get_species_data(predicted_species)
        base = self._build_base(bio_info, predicted_species, current_confidence)

        # Tính CF từ đầu dựa trên tất cả câu đã trả lời
        # (server là nguồn sự thật, không tin hoàn toàn giá trị client gửi lên)
        # Lấy pool câu hỏi đầy đủ (confidence thấp nhất để có hết câu)
        all_q = self.kb.get_adaptive_questions(predicted_species, 0.30)
        cf = current_confidence

        for q in all_q:
            qid = q["id"]
            if qid in answered:
                evidence = q["cf_yes"] if answered[qid] else q["cf_no"]
                cf = self.fuzzy.update_certainty_factor(cf, evidence)

        fuzzy_result = get_fuzzy_assessment(cf)
        base.update({
            "confidence": round(cf * 100, 2),
            "fuzzy_status": fuzzy_result["fuzzy_status"],
            "fuzzy_message": fuzzy_result["message"],
        })

        # ── ĐẠT NGƯỠNG → SUCCESS ────────────────────────────────────
        if cf >= 0.90:
            forward = self.inference.forward_chaining({"predicted_species": predicted_species})
            bio_full = forward.get("biology") or bio_info
            bio_feats = self.kb.get_identification_features(predicted_species)
            if isinstance(bio_full, dict):
                if "dac_diem_nhan_dang" not in bio_full or not bio_full["dac_diem_nhan_dang"]:
                    bio_full["dac_diem_nhan_dang"] = bio_feats
                else:
                    for k, v in bio_feats.items():
                        if v and not bio_full["dac_diem_nhan_dang"].get(k):
                            bio_full["dac_diem_nhan_dang"][k] = v
            return {
                **base,
                "status": "SUCCESS",
                "message": "Đã đủ thông tin xác nhận. Hiển thị kết quả đầy đủ.",
                "biology": bio_full,
                "legal": forward.get("legal"),
                "inferred_legal_group": forward.get("legal_group"),
            }

        # ── RỚT NGƯỠNG → REJECTED ────────────────────────────────────
        if cf <= 0.30:
            return {
                **base,
                "status": "REJECTED",
                "message": (f"Sau khi xác nhận, độ tin cậy giảm xuống {round(cf*100, 2)}%. "
                            "Loài này không có trong dữ liệu hệ thống."),
            }

        # ── VẪN TRONG VÙNG TRUNG GIAN → hỏi tiếp (câu chưa trả lời) ─
        pending_questions = self.kb.get_adaptive_questions(predicted_species, cf)
        answered_ids = set(answered.keys())
        remaining = [q for q in pending_questions if q["id"] not in answered_ids]

        if not remaining:
            return {
                **base,
                "status": "REJECTED",
                "message": (f"Đã xác nhận hết nhưng độ tin cậy vẫn chưa đủ ({round(cf*100,2)}%). "
                            "Vui lòng tham vấn chuyên gia sinh học."),
            }

        return {
            **base,
            "status": "ASKING",
            "message": f"Độ tin cậy hiện tại: {round(cf*100, 2)}%. Cần xác nhận thêm.",
            "questions": remaining,
        }