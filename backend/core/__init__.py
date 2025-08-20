# Core package
"""Core business logic and models for the PM reporting system."""

from .config import SAMPLE_STAKEHOLDER_QUESTIONS, SUPPORTED_ROLES
from .interfaces import LLMInterface
from .models import AggregatedData, Update
from .settings import settings

__all__ = [
    "LLMInterface",
    "AggregatedData",
    "Update",
    "SUPPORTED_ROLES",
    "SAMPLE_STAKEHOLDER_QUESTIONS",
    "settings",
]
