import json
import logging
from typing import List, Dict, Any
from core.rule import Rule

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Luu tru va quan ly tap luat (rules) cung nhu thong tin phap ly."""
    def __init__(self):
        self.rules: List[Rule] = []
        self.legal_db: Dict[str, Any] = {}

    def load_rules(self, filepath: str):
        """
        Load luat tu file JSON (Dinh dang IF-THEN).
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    self.add_rule(item)
            logger.info(f"Da load {len(self.rules)} luat tu {filepath}.")
        except Exception as e:
            logger.error(f"Loi khi load rules tu {filepath}: {e}")

    def load_legal_info(self, filepath: str):
        """
        Load thong tin phap ly tu file animal_rulebase.json cua ban.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.legal_db = json.load(f)
            logger.info(f"Da load thong tin phap ly tu {filepath}.")
        except Exception as e:
            logger.error(f"Loi khi load legal info tu {filepath}: {e}")

    def add_rule(self, rule_dict: dict):
        rule = Rule(
            rule_id=rule_dict.get("rule_id", f"R_AUTO_{len(self.rules)+1}"),
            premises=rule_dict["if"],
            conclusion=rule_dict["then"],
            cf=rule_dict.get("cf", 1.0)
        )
        self.rules.append(rule)
        
    def get_legal_info(self, species: str) -> dict:
        """Lay thong tin phap ly theo ten khoa hoc hoac ten thuong goi."""
        if species in self.legal_db:
            return self.legal_db[species]
        return {
            "legal_advice": "Khong tim thay thong tin phap ly cho loai nay."
        }

    def reset_rules(self):
        for r in self.rules:
            r.reset()