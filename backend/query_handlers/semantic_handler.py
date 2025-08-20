# semantic_handler.py
"""Handler for semantic search queries."""

from typing import Any, Dict, List, Tuple


class SemanticHandler:
    """Handles semantic search queries"""

    def handle(
        self, query: str, vector_service, reporting_tool, params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle semantic search queries"""

        search_limit = params.get("search_limit", 10)

        # Build search filters
        filters = {}
        if "employee_filter" in params:
            filters["employee"] = params["employee_filter"]
        if "role_filter" in params:
            filters["role"] = params["role_filter"]
        if "date_range" in params:
            filters["date_from"] = params["date_range"]["start"]
            filters["date_to"] = params["date_range"]["end"]

        # Perform semantic search
        search_results = vector_service.semantic_search(
            query, limit=search_limit, filters=filters if filters else None
        )

        if not search_results:
            # Fallback to traditional Q&A
            answer = reporting_tool.answer_factual_question_from_all_data(query)
            additional_info = {
                "method": "semantic_fallback_to_traditional",
                "search_results": 0,
            }
            return f"(No semantic matches found) {answer}", additional_info

        # Format semantic results into a comprehensive answer
        answer = self._format_semantic_results(query, search_results)

        additional_info = {
            "method": "semantic_search",
            "search_results": len(search_results),
            "top_matches": [
                {
                    "employee": result["employee"],
                    "relevance": round(result["similarity_score"], 3),
                    "snippet": result["document"][:100] + "..."
                    if len(result["document"]) > 100
                    else result["document"],
                }
                for result in search_results[:3]
            ],
        }

        return answer, additional_info

    def _format_semantic_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> str:
        """Format semantic search results into a comprehensive answer"""

        if not results:
            return "No relevant information found."

        # Sort results by similarity score (highest first)
        sorted_results = sorted(
            results, key=lambda r: r["similarity_score"], reverse=True
        )

        answer_parts = [
            f"Based on semantic analysis of team updates, here's what I found regarding: '{query}'"
        ]

        # Always show the top results regardless of similarity threshold
        answer_parts.append("\n**Most Relevant Updates:**")

        # Show top 5 results with full content
        for result in sorted_results[:5]:
            relevance_percent = result["similarity_score"] * 100
            answer_parts.append(
                f"• **{result['employee']}** ({result['role']}, {result['date']}): "
                f"{result['document']} "
                f"*(Relevance: {relevance_percent:.1f}%)*"
            )

        # If we have more results, show additional ones with truncated content
        if len(sorted_results) > 5:
            answer_parts.append("\n**Additional Related Updates:**")
            for result in sorted_results[5:8]:  # Show next 3
                relevance_percent = result["similarity_score"] * 100
                truncated_content = (
                    (result["document"][:100] + "...")
                    if len(result["document"]) > 100
                    else result["document"]
                )
                answer_parts.append(
                    f"• **{result['employee']}** ({result['role']}, {result['date']}): "
                    f"{truncated_content} "
                    f"*(Relevance: {relevance_percent:.1f}%)*"
                )

        # Add comprehensive summary
        employees = list({r["employee"] for r in results})
        roles = list({r["role"] for r in results})
        avg_relevance = sum(r["similarity_score"] for r in results) / len(results) * 100

        answer_parts.append(
            f"\n**Summary:** Found {len(results)} relevant updates from "
            f"{len(employees)} team members across {len(roles)} roles. "
            f"Average relevance: {avg_relevance:.1f}%"
        )

        return "\n".join(answer_parts)
