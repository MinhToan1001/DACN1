# src/knowledge/__init__.py
from .core.knowledge_base import KnowledgeBase
from .core.fuzzy_logic import FuzzyLogic, get_fuzzy_assessment
from .engine.inference import InferenceEngine
from .expert_system import ExpertSystem

__all__ = ['KnowledgeBase', 'FuzzyLogic', 'InferenceEngine', 'ExpertSystem']