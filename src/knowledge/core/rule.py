from typing import List, Tuple, Dict
from knowledge.core.fuzzy_logic import FactBase

class Rule:
    """Dai dien cho 1 luat suy dien gom: IF (premises), THEN (conclusion) va CF."""
    def __init__(self, rule_id: str, premises: List[str], conclusion: str, cf: float):
        self.rule_id = rule_id
        self.premises = premises
        self.conclusion = conclusion
        self.cf = max(0.0, min(1.0, float(cf)))
        self.is_fired = False

    def evaluate(self, fact_base: FactBase) -> Tuple[bool, float, Dict[str, float]]:
        """Kiem tra xem luat co the kich hoat khong dua tren FactBase hien tai."""
        if self.is_fired:
            return False, 0.0, {}

        premise_cfs = {}
        min_cf = 1.0
        for p in self.premises:
            cf_val = fact_base.get_cf(p)
            if cf_val == 0.0:
                return False, 0.0, {} # Thieu 1 dieu kien la khong xet
            premise_cfs[p] = cf_val
            if cf_val < min_cf:
                min_cf = cf_val

        return True, min_cf, premise_cfs

    def reset(self):
        self.is_fired = False