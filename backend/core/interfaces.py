# interfaces.py
from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """Abstract interface for LLM implementations"""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass
