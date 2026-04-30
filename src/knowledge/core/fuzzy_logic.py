"""
Fuzzy Logic Module - Theo Chapter 4: Uncertain & Inexact Reasoning
Tích hợp lý thuyết logic mờ của Zadeh cho Expert System
"""

import math
from typing import Tuple, Dict, List


class FuzzyLogic:
    """Hệ thống Logic Mờ cho Expert System nhận dạng loài và suy luận pháp lý"""

    def __init__(self):
        self.conf_membership = {
            "LOW":    lambda x: self.trapezoid(x, 0.0, 0.0, 0.4, 0.55),
            "MEDIUM": lambda x: self.triangle(x, 0.45, 0.65, 0.85),
            "HIGH":   lambda x: self.trapezoid(x, 0.75, 0.85, 1.0, 1.0)
        }
        self.legal_confidence_labels = ["Rất thấp", "Thấp", "Trung bình", "Cao", "Rất cao"]

    # ====================== MEMBERSHIP FUNCTIONS ======================
    @staticmethod
    def triangle(x: float, a: float, b: float, c: float) -> float:
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)

    @staticmethod
    def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1.0
        else:
            return (d - x) / (d - c)

    # ====================== FUZZIFICATION ======================
    def fuzzify_confidence(self, confidence: float) -> Dict[str, float]:
        confidence = max(0.0, min(1.0, confidence))
        return {label: round(func(confidence), 4) for label, func in self.conf_membership.items()}

    # ====================== FUZZY RULES ======================
    def apply_fuzzy_rules(self, conf_memberships: Dict[str, float],
                          bio_match: float = 1.0) -> Dict[str, float]:
        return {
            "LOW":    min(conf_memberships["LOW"], 1.0),
            "MEDIUM": min(conf_memberships["MEDIUM"], 1.0),
            "HIGH":   min(conf_memberships["HIGH"], bio_match)
        }

    # ====================== DEFUZZIFICATION ======================
    def defuzzify(self, fuzzy_outputs: Dict[str, float]) -> Tuple[str, float, str]:
        labels = ["LOW", "MEDIUM", "HIGH"]
        values = {"LOW": 0.3, "MEDIUM": 0.65, "HIGH": 0.92}

        numerator = sum(fuzzy_outputs[lbl] * values[lbl] for lbl in labels)
        denominator = sum(fuzzy_outputs[lbl] for lbl in labels)
        crisp_value = numerator / denominator if denominator > 0 else 0.5

        if crisp_value >= 0.85:
            status = "HIGH"
            message = "Hệ thống rất tự tin. Khuyến nghị tin tưởng kết quả và tiến hành thủ tục pháp lý."
        elif crisp_value >= 0.55:
            status = "MEDIUM"
            message = "Độ tin cậy trung bình. Nên xác nhận thêm đặc điểm sinh học trước khi kết luận pháp lý."
        else:
            status = "LOW"
            message = "Độ tin cậy thấp. Cần kiểm tra lại mẫu vật hoặc thu thập thêm dữ liệu."

        return status, round(crisp_value, 4), message

    # ====================== MAIN FUZZY ASSESSMENT ======================
    def fuzzy_assessment(self, model_confidence: float, bio_feature_match: float = 1.0) -> Dict:
        conf_mf = self.fuzzify_confidence(model_confidence)
        rule_outputs = self.apply_fuzzy_rules(conf_mf, bio_feature_match)
        status, crisp_conf, message = self.defuzzify(rule_outputs)
        return {
            "fuzzy_status": status,
            "crisp_confidence": crisp_conf,
            "message": message,
            "membership_degrees": conf_mf,
            "rule_outputs": rule_outputs,
            "bio_feature_match_used": bio_feature_match
        }

    # ====================== CERTAINTY FACTOR (MYCIN) ======================
    @staticmethod
    def update_certainty_factor(cf_old: float, cf_evidence: float) -> float:
        """
        Cập nhật niềm tin theo MYCIN Certainty Factor.
        cf_old: [0.0, 1.0]  →  cf_evidence: [-1.0, 1.0]
        """
        cf1 = (cf_old * 2) - 1.0
        cf2 = cf_evidence

        if cf1 >= 0 and cf2 >= 0:
            cf_new = cf1 + cf2 * (1 - cf1)
        elif cf1 < 0 and cf2 < 0:
            cf_new = cf1 + cf2 * (1 + cf1)
        else:
            cf_new = (cf1 + cf2) / (1 - min(abs(cf1), abs(cf2)))

        final_cf = (cf_new + 1.0) / 2.0
        return round(max(0.01, min(0.99, final_cf)), 4)


# Instance dùng chung
fuzzy_logic_engine = FuzzyLogic()


def get_fuzzy_assessment(model_conf: float, bio_match: float = 1.0):
    fuzzy = FuzzyLogic()
    return fuzzy.fuzzy_assessment(model_conf, bio_match)