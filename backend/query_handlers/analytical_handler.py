# analytical_handler.py
"""Handler for analytical queries that require deeper reasoning."""

from typing import Any, Dict, Tuple


class AnalyticalHandler:
    """Handles analytical queries that require deeper reasoning"""

    def handle(
        self, query: str, reporting_tool, vector_service, params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle analytical queries that require deeper reasoning"""
        # params reserved for future filtering/configuration options

        # For analytical queries, combine both approaches
        # First get semantic context
        semantic_results = vector_service.semantic_search(query, limit=5)

        # Then use traditional Q&A with enhanced context
        if semantic_results:
            context_updates = [
                f"{result['employee']} ({result['role']}): {result['document']}"
                for result in semantic_results
            ]
            enhanced_query = f"""
            {query}

            Additional context from similar updates:
            {chr(10).join(context_updates)}

            Please provide a comprehensive analytical response.
            """

            # Use scalable contextual query with enhanced context
            answer = reporting_tool.answer_contextual_question(
                enhanced_query,
                max_updates=75  # Leave room for semantic context
            )
        else:
            # Use scalable contextual query
            answer = reporting_tool.answer_contextual_question(
                query,
                max_updates=100
            )

        additional_info = {
            "method": "analytical_hybrid",
            "semantic_context_used": len(semantic_results) > 0,
            "context_updates": len(semantic_results),
        }

        return answer, additional_info
