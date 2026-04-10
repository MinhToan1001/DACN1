from typing import Dict

class FactBase:
    """Quan ly danh sach cac su kien (facts) hien tai va do tin cay (CF) cua chung."""
    def __init__(self):
        self.facts: Dict[str, float] = {}

    def add_fact(self, name: str, cf: float):
        """
        Them su kien. Neu da ton tai, ket hop CF.
        Cong thuc: CF_combined = CF1 + CF2 * (1 - CF1)
        """
        cf = max(0.0, min(1.0, float(cf)))
        
        if name in self.facts:
            old_cf = self.facts[name]
            new_cf = old_cf + cf * (1.0 - old_cf)
            self.facts[name] = new_cf
        else:
            self.facts[name] = cf

    def get_cf(self, name: str) -> float:
        return self.facts.get(name, 0.0)

    def has_fact(self, name: str) -> bool:
        return self.get_cf(name) > 0.0
        
    def get_all(self) -> Dict[str, float]:
        return self.facts.copy()

    def clear(self):
        self.facts.clear()