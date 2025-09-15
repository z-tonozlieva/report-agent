# smart_router.py
"""Main orchestrating router that uses the modular query handling system."""

import logging
from typing import Optional

from backend.core import LLMInterface
from .analytical_handler import AnalyticalHandler
from .classifier import QueryClassifier
from .comparison_handler import ComparisonHandler
from .extractor import EntityExtractor
from .factual_handler import FactualHandler
from .report_handler import ReportHandler
from .semantic_handler import SemanticHandler
from .types import QueryClassification, RouterResponse

logger = logging.getLogger(__name__)


class SmartQueryRouter:
    """Intelligent query router that directs queries to the most appropriate backend"""

    def __init__(self, llm: Optional[LLMInterface] = None):
        self.llm = llm

        # Initialize components
        self.entity_extractor = EntityExtractor()
        self.query_classifier = QueryClassifier(self.entity_extractor)

        # Initialize handlers
        self.report_handler = ReportHandler()
        self.semantic_handler = SemanticHandler()
        self.analytical_handler = AnalyticalHandler()
        self.factual_handler = FactualHandler()
        self.comparison_handler = ComparisonHandler()

    def route_query(
        self,
        query: str,
        reporting_tool,
        vector_service,
        db_session=None,
        override_method: Optional[str] = None,
    ) -> RouterResponse:
        """Route a query to the most appropriate backend and return response"""

        # Update dynamic patterns from current data
        self.update_dynamic_patterns(reporting_tool)

        # Classify query unless method is overridden
        if override_method:
            classification = QueryClassification(
                query_type=override_method,
                confidence=1.0,
                reasoning=f"Method overridden to {override_method}",
                suggested_params={},
            )
        else:
            classification = self.query_classifier.classify_query(query)

        logger.info(
            f"Query classified as '{classification.query_type}' with confidence {classification.confidence:.2f}"
        )

        try:
            # Route to appropriate handler
            if classification.query_type == "report":
                answer, additional_info = self.report_handler.handle(
                    query, reporting_tool, classification.suggested_params
                )
            elif classification.query_type == "semantic":
                answer, additional_info = self.semantic_handler.handle(
                    query,
                    vector_service,
                    reporting_tool,
                    classification.suggested_params,
                )
            elif classification.query_type == "analytical":
                answer, additional_info = self.analytical_handler.handle(
                    query,
                    reporting_tool,
                    vector_service,
                    classification.suggested_params,
                )
            elif classification.query_type == "comparison":
                answer, additional_info = self.comparison_handler.handle(
                    query,
                    reporting_tool,
                    vector_service,
                    classification.suggested_params,
                    classification.entities,
                )
            else:  # factual
                answer, additional_info = self.factual_handler.handle(
                    query, reporting_tool, classification.suggested_params
                )

            return RouterResponse(
                answer=answer,
                method_used=classification.query_type,
                confidence=classification.confidence,
                additional_info={
                    "classification": classification,
                    "routing_info": additional_info,
                },
            )

        except Exception as e:
            logger.error(f"Error routing query: {str(e)}")
            # Fallback to simple contextual query
            try:
                answer = reporting_tool.answer_contextual_question(query, max_updates=50)
                return RouterResponse(
                    answer=f"(Fallback) {answer}",
                    method_used="contextual_fallback",
                    confidence=0.5,
                    additional_info={"error": str(e)},
                )
            except Exception as fallback_error:
                return RouterResponse(
                    answer=f"Unable to process query: {str(fallback_error)}",
                    method_used="error",
                    confidence=0.0,
                    additional_info={"error": str(fallback_error)},
                )

    def update_dynamic_patterns(self, reporting_tool) -> None:
        """Update patterns from current reporting tool data"""
        self.entity_extractor.update_cache(reporting_tool)

    # Legacy method for backwards compatibility
    def classify_query(self, query: str) -> QueryClassification:
        """Classify a query (legacy compatibility method)"""
        return self.query_classifier.classify_query(query)

    # Legacy method for backwards compatibility
    def extract_entities(self, query: str):
        """Extract entities (legacy compatibility method)"""
        return self.entity_extractor.extract_entities(query)
