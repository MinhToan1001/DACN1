class InferenceEngine:
    """Forward/backward inference engine with visible trace logging."""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def _trace(self, steps, message: str):
        print(f"[INFERENCE] {message}")
        if steps is not None:
            steps.append(message)

    def forward_chaining(self, initial_facts: dict, trace_steps=None):
        """Forward chaining: facts -> biology and legal conclusions."""
        predicted_species = initial_facts.get("predicted_species")
        self._trace(trace_steps, f"Forward chaining started with facts={initial_facts}")
        if not predicted_species:
            self._trace(trace_steps, "Missing predicted_species, stop inference.")
            return {"error": "No predicted species provided"}

        self._trace(trace_steps, f"Rule R1: predicted_species -> lookup species data for '{predicted_species}'")
        bio_info, legal_info = self.kb.get_species_data(predicted_species)
        self._trace(trace_steps, f"Knowledge lookup: biology_found={bool(bio_info)}, legal_found={bool(legal_info)}")
        legal_group = (
            legal_info.get("nhom_phap_ly") or
            legal_info.get("legal_group") or
            bio_info.get("nhom_phap_ly", "Chua xac dinh")
        )
        self._trace(trace_steps, f"Rule R2: legal group inferred as '{legal_group}'")

        result = {
            "predicted_species": predicted_species,
            "biology": bio_info,
            "legal": legal_info,
            "legal_group": legal_group,
            "legal_status": f"Nhom phap ly: {legal_group} - {legal_info.get('mo_ta', 'Khong co mo ta phap ly')}",
        }
        self._trace(trace_steps, "Forward chaining completed: biology + legal conclusions are ready.")
        return result

    def backward_chaining(self, goal: str, initial_facts: dict = None, trace_steps=None):
        """Backward chaining: generate adaptive confirmation questions."""
        if initial_facts is None:
            initial_facts = {}

        self._trace(trace_steps, f"Backward chaining started for goal='{goal}', facts={initial_facts}")
        predicted_species = initial_facts.get("predicted_species")
        if not predicted_species:
            self._trace(trace_steps, "Missing predicted_species, no confirmation question generated.")
            return {"questions": []}

        current_conf = initial_facts.get("current_confidence", 0.60)
        questions = self.kb.get_adaptive_questions(predicted_species, current_conf)
        self._trace(trace_steps, f"Generated {len(questions)} adaptive question(s) at confidence={current_conf:.4f}.")
        return {"questions": questions}
