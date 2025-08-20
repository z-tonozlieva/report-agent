# query_types.py
"""Data types and models for the query routing system."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set


@dataclass
class EntityExtraction:
    """Extracted entities from a query"""

    projects: Set[str]
    technologies: Set[str]
    team_names: Set[str]
    focus_areas: Set[str]
    comparison_terms: Set[str]
    temporal_expressions: Set[str]


@dataclass
class QueryClassification:
    """Classification result for a query"""

    query_type: str  # 'report', 'factual', 'semantic', 'analytical', 'comparison'
    confidence: float
    reasoning: str
    suggested_params: Dict[str, Any]
    entities: Optional[EntityExtraction] = None


@dataclass
class RouterResponse:
    """Response from the query router"""

    answer: str
    method_used: str
    confidence: float
    additional_info: Dict[str, Any] = None
