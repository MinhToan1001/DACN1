import logging
from typing import Dict, Any, List
from core.knowledge_base import KnowledgeBase
from core.fact_base import FactBase

logger = logging.getLogger(__name__)

class InferenceEngine:
    """He thong suy dien tien (Forward Chaining) su dung CF."""
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.fact_base = FactBase()
        self.execution_log: List[str] = []
        self.THRESHOLD = 0.8

    def calculate_cf(self, min_premise_cf: float, rule_cf: float) -> float:
        return min_premise_cf * rule_cf

    def _log_explanation(self, rule, premise_cfs: dict, derived_cf: float):
        log_lines = [f"Rule {rule.rule_id} activated:"]
        if_parts = [f"{k} ({v:.2f})" for k, v in premise_cfs.items()]
        log_lines.append(f"  IF: {', '.join(if_parts)}")
        log_lines.append(f"  THEN: {rule.conclusion}")
        min_cf_str = f"min({', '.join([f'{v:.2f}' for v in premise_cfs.values()])})"
        log_lines.append(f"  CF = {min_cf_str} * {rule.cf:.2f} = {derived_cf:.4f}")
        self.execution_log.append("\n".join(log_lines))

    def load_dl_predictions(self, dl_output: Dict[str, float]):
        for species, prob in dl_output.items():
            if prob > 0:
                self.fact_base.add_fact(species, prob)
                self.execution_log.append(f"DL Integration: Them fact '{species}' (CF={prob:.4f})")

    def infer(self, user_facts: Dict[str, float], debug: bool = False) -> Dict[str, Any]:
        self.kb.reset_rules()
        self.execution_log.clear()
        
        # Load facts
        for fact, cf in user_facts.items():
            self.fact_base.add_fact(fact, cf)
            if debug:
                self.execution_log.append(f"Input Fact: {fact} (CF={cf:.2f})")

        new_facts_derived = True
        
        # Forward Chaining Loop
        while new_facts_derived:
            new_facts_derived = False
            for rule in self.kb.rules:
                can_fire, min_cf, premise_cfs = rule.evaluate(self.fact_base)
                if can_fire:
                    derived_cf = self.calculate_cf(min_cf, rule.cf)
                    self.fact_base.add_fact(rule.conclusion, derived_cf)
                    self._log_explanation(rule, premise_cfs, derived_cf)
                    rule.is_fired = True
                    new_facts_derived = True

        # Trich xuat ket qua (So khop FactBase voi Legal_DB)
        results = []
        for fact_name, final_cf in self.fact_base.get_all().items():
            if fact_name in self.kb.legal_db:
                results.append({
                    "species": fact_name,
                    "confidence": round(final_cf, 4),
                    "legal": self.kb.get_legal_info(fact_name)
                })

        results = sorted(results, key=lambda x: x["confidence"], reverse=True)

        final_response = {
            "status": "success",
            "needs_more_info": False,
            "predictions": results,
            "top_prediction": results[0] if results else None
        }

        if results and results[0]["confidence"] < self.THRESHOLD:
            final_response["needs_more_info"] = True
            final_response["status"] = f"CF = {results[0]['confidence']} < {self.THRESHOLD}. Can them thong tin."
        elif not results:
            final_response["status"] = "Khong suy dien ra ket qua nao tuong ung trong DB."
            final_response["needs_more_info"] = True

        return final_response

    def explain(self) -> str:
        if not self.execution_log:
            return "Khong co luat nao duoc kich hoat."
        return "\n\n".join(self.execution_log)