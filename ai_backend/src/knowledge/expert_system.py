import unicodedata

from .core.knowledge_base import KnowledgeBase
from .core.fuzzy_logic import FuzzyLogic, get_fuzzy_assessment
from .engine.inference import InferenceEngine


class ExpertSystem:
    """Expert system: fuzzy logic + MYCIN CF + forward/backward inference."""

    def __init__(self, rules_path: str):
        self.kb = KnowledgeBase(rules_path)
        self.inference = InferenceEngine(self.kb)
        self.fuzzy = FuzzyLogic()

    def _trace(self, steps: list, message: str, phase: str = "Suy diễn", level: str = "info"):
        print(f"[EXPERT] [{phase}] {message}")
        steps.append({
            "phase": phase,
            "level": level,
            "message": message,
        })

    @staticmethod
    def _with_trace(payload: dict, steps: list) -> dict:
        payload["inference_trace"] = steps
        return payload

    def _build_base(self, bio_info: dict, predicted_species: str, confidence_raw: float) -> dict:
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

    def _complete_biology(self, predicted_species: str, bio_info: dict, forward: dict) -> dict:
        bio_full = forward.get("biology") or bio_info
        bio_feats = self.kb.get_identification_features(predicted_species)
        if isinstance(bio_full, dict):
            if "dac_diem_nhan_dang" not in bio_full or not bio_full["dac_diem_nhan_dang"]:
                bio_full["dac_diem_nhan_dang"] = bio_feats
            else:
                for k, v in bio_feats.items():
                    if v and not bio_full["dac_diem_nhan_dang"].get(k):
                        bio_full["dac_diem_nhan_dang"][k] = v
        return bio_full

    def initial_predict(self, predicted_species: str, raw_confidence: float) -> dict:
        trace_steps = []
        self._trace(trace_steps, f"Dự đoán ban đầu: loài='{predicted_species}', CF mô hình={raw_confidence:.4f}")

        fuzzy_preview = get_fuzzy_assessment(raw_confidence)
        self._trace(
            trace_steps,
            f"Mờ hóa độ tin cậy: trạng thái={fuzzy_preview['fuzzy_status']}, giá trị rõ={fuzzy_preview['crisp_confidence']}",
        )

        bio_info, legal_info = self.kb.get_species_data(predicted_species)
        self._trace(trace_steps, f"Tra cứu tri thức: có dữ liệu sinh học={bool(bio_info)}, có dữ liệu pháp lý={bool(legal_info)}")
        base = self._build_base(bio_info, predicted_species, raw_confidence)

        if raw_confidence >= 0.90:
            self._trace(trace_steps, "Độ tin cậy >= 90%, chấp nhận kết quả và suy diễn pháp lý.", "Kết luận", "success")
            forward = self.inference.forward_chaining({"predicted_species": predicted_species}, trace_steps)
            bio_full = self._complete_biology(predicted_species, bio_info, forward)
            return self._with_trace({
                **base,
                "status": "SUCCESS",
                "message": "Hệ thống có độ tin cậy cao với dự đoán này.",
                "biology": bio_full,
                "legal": forward.get("legal"),
                "inferred_legal_group": forward.get("legal_group"),
            }, trace_steps)

        if raw_confidence <= 0.30:
            self._trace(trace_steps, "Độ tin cậy <= 30%, từ chối kết quả nhận dạng.", "Kết luận", "danger")
            return self._with_trace({
                **base,
                "status": "REJECTED",
                "message": (
                    f"Độ tin cậy quá thấp ({round(raw_confidence * 100, 2)}%). "
                    "Loài này chưa có trong dữ liệu hệ thống hoặc không thể nhận dạng."
                ),
            }, trace_steps)

        self._trace(trace_steps, "Độ tin cậy chưa đạt ngưỡng, cần hỏi thêm đặc điểm nhận dạng.", "Hỏi đáp", "warning")
        backward = self.inference.backward_chaining(
            "confirm_species",
            {"predicted_species": predicted_species, "current_confidence": raw_confidence},
            trace_steps,
        )
        question_to_ask = (backward.get("questions") or [])[:1]
        self._trace(trace_steps, f"Đã chọn {len(question_to_ask)} câu hỏi tiếp theo để người dùng xác nhận.")

        return self._with_trace({
            **base,
            "status": "ASKING",
            "message": (
                f"Độ tin cậy {round(raw_confidence * 100, 2)}% chưa đạt ngưỡng 90%. "
                "Vui lòng xác nhận đặc điểm sau:"
            ),
            "questions": question_to_ask,
        }, trace_steps)

    def process_answer(self, predicted_species: str, current_confidence: float, answered: dict) -> dict:
        trace_steps = []
        self._trace(
            trace_steps,
            f"Xử lý câu trả lời: loài='{predicted_species}', CF đầu vào={current_confidence:.4f}, câu trả lời={answered}",
        )

        bio_info, legal_info = self.kb.get_species_data(predicted_species)
        self._trace(trace_steps, f"Tra cứu tri thức: có dữ liệu sinh học={bool(bio_info)}, có dữ liệu pháp lý={bool(legal_info)}")
        base = self._build_base(bio_info, predicted_species, current_confidence)

        all_q = self.kb.get_adaptive_questions(predicted_species, 0.30)
        cf = current_confidence
        self._trace(trace_steps, f"Đã tải {len(all_q)} câu hỏi bằng chứng. Tính lại hệ số chắc chắn MYCIN.")

        for q in all_q:
            qid = q["id"]
            if qid in answered:
                answer_value = answered[qid]
                if answer_value == "unknown":
                    evidence = q.get("cf_unknown", 0)
                else:
                    evidence = q["cf_yes"] if bool(answer_value) else q["cf_no"]
                old_cf = cf
                cf = self.fuzzy.update_certainty_factor(cf, evidence)
                self._trace(
                    trace_steps,
                    f"Bằng chứng '{qid}'={answer_value}: CF cũ={old_cf:.4f}, tác động={evidence:.4f}, CF mới={cf:.4f}",
                    "Cập nhật CF",
                )

        fuzzy_result = get_fuzzy_assessment(cf)
        self._trace(trace_steps, f"Đánh giá mờ lại: trạng thái={fuzzy_result['fuzzy_status']}, giá trị rõ={fuzzy_result['crisp_confidence']}")
        base.update({
            "confidence": round(cf * 100, 2),
            "fuzzy_status": fuzzy_result["fuzzy_status"],
            "fuzzy_message": fuzzy_result["message"],
        })

        if cf >= 0.90:
            self._trace(trace_steps, "CF sau xác nhận >= 90%, chấp nhận và suy diễn pháp lý.", "Kết luận", "success")
            forward = self.inference.forward_chaining({"predicted_species": predicted_species}, trace_steps)
            bio_full = self._complete_biology(predicted_species, bio_info, forward)
            return self._with_trace({
                **base,
                "status": "SUCCESS",
                "message": "Đã đủ thông tin xác nhận. Hiển thị kết quả đầy đủ.",
                "biology": bio_full,
                "legal": forward.get("legal"),
                "inferred_legal_group": forward.get("legal_group"),
            }, trace_steps)

        if cf <= 0.30:
            self._trace(trace_steps, "CF sau xác nhận <= 30%, từ chối kết quả.", "Kết luận", "danger")
            return self._with_trace({
                **base,
                "status": "REJECTED",
                "message": (
                    f"Sau khi xác nhận, độ tin cậy giảm xuống {round(cf * 100, 2)}%. "
                    "Loài này không có trong dữ liệu hệ thống."
                ),
            }, trace_steps)

        pending_questions = self.kb.get_adaptive_questions(predicted_species, cf)
        answered_ids = set(answered.keys())
        remaining = [q for q in pending_questions if q["id"] not in answered_ids]
        self._trace(trace_steps, f"Vẫn chưa chắc chắn: còn {len(remaining)} câu hỏi.")

        if not remaining:
            if 0.70 <= cf < 0.90:
                self._trace(trace_steps, "Hết câu hỏi, CF nằm trong khoảng 70%-90% nên chưa kết luận chắc chắn.", "Kết luận", "warning")
                return self._with_trace({
                    **base,
                    "status": "UNCERTAIN",
                    "message": (
                        f"Loài của bạn trông rất giống loài {base.get('vietnamese_name')} "
                        f"({predicted_species}) với độ tin cậy {round(cf * 100, 2)}%, "
                        "nhưng có một số đặc điểm không trùng khớp hoặc chưa đủ thông tin nên tôi chưa thể kết luận. "
                        "Bạn có thể tìm hiểu thêm hoặc liên hệ chuyên gia/kiểm lâm địa phương để được xác minh."
                    ),
                }, trace_steps)

            self._trace(trace_steps, "Hết câu hỏi nhưng CF vẫn dưới 70%, không đủ cơ sở kết luận.", "Kết luận", "danger")
            return self._with_trace({
                **base,
                "status": "REJECTED",
                "message": (
                    f"Đã xác nhận hết chi tiết nhưng độ tin cậy vẫn chỉ đạt {round(cf * 100, 2)}% "
                    "(cần 90%). Hệ thống không đủ tự tin để kết luận pháp lý."
                ),
            }, trace_steps)

        next_question = remaining[:1]
        self._trace(trace_steps, f"Đã chọn câu hỏi tiếp theo: '{next_question[0]['id']}'.")
        return self._with_trace({
            **base,
            "status": "ASKING",
            "message": f"Độ tin cậy hiện tại: {round(cf * 100, 2)}%. Vui lòng xác nhận tiếp:",
            "questions": next_question,
        }, trace_steps)

    @staticmethod
    def _legal_advice(legal_info: dict) -> dict:
        return legal_info.get("legal_advice") if isinstance(legal_info.get("legal_advice"), dict) else legal_info

    @staticmethod
    def _display_title(key: str) -> str:
        return key.replace("_", " ").strip()

    @staticmethod
    def _normalize_legal_group(legal_group: str) -> str:
        group = (legal_group or "").upper()
        group = unicodedata.normalize("NFD", group)
        group = "".join(ch for ch in group if unicodedata.category(ch) != "Mn")
        group = group.replace("NHOM", "").replace("NHÓM", "").strip()
        if "IIB" in group or "II B" in group or "2B" in group or "II-B" in group:
            return "IIB"
        if "IB" in group or "I B" in group or "1B" in group or "I-B" in group:
            return "IB"
        if "THONG" in group or "THƯỜNG" in group or "THUONG" in group:
            return "THONG_THUONG"
        return group

    @staticmethod
    def _ordinary_captivity_conditions() -> list:
        return [
            "Đảm bảo nguồn gốc động vật rừng nuôi hợp pháp theo quy định của pháp luật.",
            "Đảm bảo an toàn cho con người; thực hiện các quy định của pháp luật về môi trường, thú y.",
            "Ghi chép sổ theo dõi vật nuôi theo Mẫu số 16 tại Phụ lục ban hành kèm Nghị định 06/2019/NĐ-CP.",
            "Trong thời hạn tối đa 03 ngày làm việc kể từ ngày đưa động vật rừng thông thường về cơ sở nuôi, phải gửi thông báo cho cơ quan Kiểm lâm sở tại để theo dõi, quản lý.",
        ]

    def _ordinary_legal_summary(self, legal_group: str, advice: dict) -> dict:
        items = advice.get("ordinary_captivity_conditions")
        if not isinstance(items, list) or not items:
            items = self._ordinary_captivity_conditions()
        return {
            "id": "dieu_11_nghi_dinh_06_2019_nd_cp",
            "title": "Điều kiện nuôi động vật rừng thông thường",
            "category": "Điều 11 Nghị định 06/2019/NĐ-CP",
            "items": [str(item) for item in items],
        }

    def _collect_penalty_frames(self, legal_info: dict) -> list:
        advice = self._legal_advice(legal_info or {})
        frames = []
        for source_key, category in (
            ("criminal_penalties", "Hình sự"),
            ("administrative_penalties", "Hành chính"),
            ("additional_penalties", "Bổ sung"),
            ("remedial_measures", "Khắc phục"),
            ("additional_and_remedial", "Bổ sung và khắc phục"),
        ):
            source = advice.get(source_key)
            if not source:
                continue
            if isinstance(source, dict):
                for key, value in source.items():
                    items = value if isinstance(value, list) else [value]
                    frames.append({
                        "id": key,
                        "title": self._display_title(key),
                        "category": category,
                        "items": [str(item) for item in items],
                    })
            elif isinstance(source, list):
                frames.append({
                    "id": source_key,
                    "title": self._display_title(source_key),
                    "category": category,
                    "items": [str(item) for item in source],
                })
        return frames

    @staticmethod
    def _choose_frame(frames: list, legal_group: str, quantity: int):
        group = (legal_group or "").upper()
        quantity = max(0, int(quantity or 0))

        if "IB" in group or group == "1":
            if quantity >= 12:
                preferred = ["KHUNG_3", "TRUY_CUU"]
            elif quantity >= 8:
                preferred = ["KHUNG_2"]
            else:
                preferred = ["KHUNG_1", "TRUY_CUU"]
        else:
            idx = min(max(quantity, 1), 11)
            preferred = [f"KHUNG_{idx}", f"KHUNG_{idx}_", f"KHUNG_{idx}A"]

        for token in preferred:
            for frame in frames:
                if token in frame["id"].upper():
                    return frame
        return frames[0] if frames else None

    def infer_legal_scenario(self, predicted_species: str, possession_status: str, quantity: int = 0) -> dict:
        trace_steps = []
        status = (possession_status or "observe").strip().lower()
        self._trace(trace_steps, f"Suy diễn tình huống pháp lý: loài='{predicted_species}', trạng thái='{status}', số lượng={quantity}")

        bio_info, legal_info = self.kb.get_species_data(predicted_species)
        legal_group = (
            legal_info.get("nhom_phap_ly") or
            legal_info.get("legal_group") or
            bio_info.get("nhom_phap_ly", "Chưa xác định")
        )
        advice = self._legal_advice(legal_info or {})
        frames = self._collect_penalty_frames(legal_info or {})
        normalized_group = self._normalize_legal_group(legal_group)
        self._trace(trace_steps, f"Nhóm pháp lý='{legal_group}' -> chuẩn hóa='{normalized_group}', số khung={len(frames)}", "Pháp lý")

        if normalized_group == "THONG_THUONG":
            ordinary_frame = self._ordinary_legal_summary(legal_group, advice)
            self._trace(trace_steps, "Động vật rừng thông thường: chỉ hiển thị điều kiện nuôi theo Điều 11, không gán khung phạt IIB.", "Pháp lý", "success")
            return self._with_trace({
                "status": "ORDINARY_CAPTIVITY" if status != "observe" else "ORDINARY_OBSERVE",
                "legal_group": legal_group,
                "group_name": advice.get("group_name", legal_group),
                "quantity": int(quantity or 0),
                "message": (
                    "Đây là động vật rừng thông thường. Nếu nuôi, tổ chức/cá nhân cần đáp ứng các điều kiện "
                    "tại Điều 11 Nghị định 06/2019/NĐ-CP; hệ thống không áp dụng các khung xử phạt dành cho nhóm IIB."
                ),
                "selected_frame": ordinary_frame,
                "all_frames": [],
                "captivity_conditions": ordinary_frame["items"],
            }, trace_steps)

        if status == "captivity" and normalized_group not in {"IIB", "IB"}:
            self._trace(trace_steps, "Không phải nhóm IIB/IB, không tự động chọn các khung phạt nuôi/nhốt theo số lượng.", "Pháp lý", "warning")

        if status == "observe":
            self._trace(trace_steps, "Người dùng chỉ quan sát: hiển thị thông tin pháp lý tham khảo.", "Pháp lý")
            return self._with_trace({
                "status": "OBSERVE",
                "legal_group": legal_group,
                "group_name": advice.get("group_name", legal_group),
                "message": "Bạn chỉ quan sát/nhìn thấy loài này. Hệ thống hiển thị toàn bộ thông tin pháp lý để tham khảo.",
                "selected_frame": None,
                "all_frames": frames,
            }, trace_steps)

        selected = self._choose_frame(frames, legal_group, quantity) if normalized_group in {"IIB", "IB"} else None
        self._trace(trace_steps, f"Người dùng đang nuôi/giữ: khung được chọn='{selected['id'] if selected else None}'", "Pháp lý")
        return self._with_trace({
            "status": "CAPTIVITY",
            "legal_group": legal_group,
            "group_name": advice.get("group_name", legal_group),
            "quantity": int(quantity or 0),
            "message": (
                "Suy diễn khung pháp lý theo tình huống người dùng đang nuôi/giữ cá thể."
                if normalized_group in {"IIB", "IB"}
                else "Loài này không thuộc nhóm IIB/IB, hệ thống không tự động gán các khung phạt nuôi/nhốt theo số lượng."
            ),
            "selected_frame": selected,
            "all_frames": frames if normalized_group in {"IIB", "IB"} else [],
        }, trace_steps)
