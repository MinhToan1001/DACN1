class InferenceEngine:
    """Khung suy diễn tiến (forward) và suy diễn lùi (backward)"""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def forward_chaining(self, initial_facts: dict):
        """Suy diễn tiến: từ facts → kết luận (sinh học + pháp lý)"""
        predicted_species = initial_facts.get("predicted_species")
        if not predicted_species:
            return {"error": "No predicted species provided"}

        bio_info, legal_info = self.kb.get_species_data(predicted_species)
        legal_group = (
            legal_info.get("nhom_phap_ly") or
            legal_info.get("legal_group") or
            bio_info.get("nhom_phap_ly", "Chưa xác định")
        )

        return {
            "predicted_species": predicted_species,
            "biology": bio_info,
            "legal": legal_info,
            "legal_group": legal_group,
            "legal_status": f"Nhóm pháp lý: {legal_group} - {legal_info.get('mo_ta', 'Không có mô tả pháp lý')}",
        }

    def backward_chaining(self, goal: str, initial_facts: dict = None):
        """Suy diễn lùi: sinh câu hỏi xác nhận (dùng KB thích nghi)"""
        if initial_facts is None:
            initial_facts = {}

        predicted_species = initial_facts.get("predicted_species")
        if not predicted_species:
            return {"questions": []}

        current_conf = initial_facts.get("current_confidence", 0.60)
        questions = self.kb.get_adaptive_questions(predicted_species, current_conf)
        return {"questions": questions}