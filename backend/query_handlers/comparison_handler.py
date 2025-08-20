# comparison_handler.py
"""Handler for comparison queries that compare teams, technologies, time periods, etc."""

from typing import Any, Dict, Optional, Tuple

from .types import EntityExtraction


class ComparisonHandler:
    """Handles comparison queries that compare teams, technologies, time periods, etc."""

    def handle(
        self,
        query: str,
        reporting_tool,
        vector_service,
        params: Dict[str, Any],
        entities: Optional[EntityExtraction] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle comparison queries that compare teams, technologies, time periods, etc."""

        if not entities:
            # Fallback - could use analytical handler here
            return self._fallback_to_analytical(
                query, reporting_tool, vector_service, params
            )

        # Identify what's being compared
        comparison_context = []
        comparison_filters = []

        # Team comparison
        if len(entities.team_names) > 1:
            team_list = list(entities.team_names)
            comparison_context.append(f"Comparing teams: {', '.join(team_list)}")

            # Get updates for each team
            team_data = {}
            for team in team_list:
                team_updates = [
                    update
                    for update in reporting_tool.updates
                    if team.lower() in update.update.lower()
                    or team.lower() in update.role.lower()
                ]
                team_data[team] = team_updates

            comparison_filters.append({"type": "team", "data": team_data})

        # Technology comparison
        if len(entities.technologies) > 1:
            tech_list = list(entities.technologies)
            comparison_context.append(f"Comparing technologies: {', '.join(tech_list)}")

            # Get updates mentioning each technology
            tech_data = {}
            for tech in tech_list:
                tech_updates = [
                    update
                    for update in reporting_tool.updates
                    if tech.lower() in update.update.lower()
                ]
                tech_data[tech] = tech_updates

            comparison_filters.append({"type": "technology", "data": tech_data})

        # Time period comparison (if temporal expressions suggest comparison)
        if entities.temporal_expressions and any(
            "ago" in expr for expr in entities.temporal_expressions
        ):
            comparison_context.append("Comparing time periods")

        # Role comparison
        if entities.comparison_terms:
            terms = list(entities.comparison_terms)
            comparison_context.append(f"Analyzing comparison terms: {', '.join(terms)}")

        # Build comparison prompt
        if comparison_context:
            context_description = "; ".join(comparison_context)
        else:
            context_description = "General comparison analysis"

        # Get relevant data using semantic search for context
        semantic_results = vector_service.semantic_search(query, limit=15)

        # Create enhanced prompt for comparison
        if semantic_results:
            context_updates = []
            for result in semantic_results[:10]:  # Top 10 for context
                context_updates.append(
                    f"{result['employee']} ({result['role']}, {result['date']}): {result['document']}"
                )

            comparison_prompt = f"""
            {query}
            
            Context: {context_description}
            
            Please provide a detailed comparison analysis based on the following team updates:
            
            {chr(10).join(context_updates)}
            
            Structure your response to clearly compare the requested items, highlighting:
            - Key differences and similarities
            - Performance or progress indicators
            - Specific examples from the data
            - Trends or patterns
            
            Provide a comprehensive comparison analysis.
            """
        else:
            # Fallback to all data analysis
            all_updates_context = "\n".join(
                [
                    f"{update.employee} ({update.role}, {update.date}): {update.update}"
                    for update in reporting_tool.updates[-20:]  # Last 20 updates
                ]
            )

            comparison_prompt = f"""
            {query}
            
            Context: {context_description}
            
            Based on the following team updates, provide a detailed comparison analysis:
            
            {all_updates_context}
            
            Please structure your response to clearly compare the requested items.
            """

        answer = reporting_tool.llm.generate_response(comparison_prompt)

        additional_info = {
            "method": "comparison_analysis",
            "comparison_context": comparison_context,
            "entities_found": {
                "teams": list(entities.team_names) if entities.team_names else [],
                "technologies": list(entities.technologies)
                if entities.technologies
                else [],
                "comparison_terms": list(entities.comparison_terms)
                if entities.comparison_terms
                else [],
            },
            "semantic_context_used": len(semantic_results) > 0,
            "comparison_filters": len(comparison_filters),
        }

        return answer, additional_info

    def _fallback_to_analytical(
        self, query: str, reporting_tool, vector_service, params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Fallback to analytical query when entities are not available"""
        # Import here to avoid circular imports
        from .analytical_handler import AnalyticalHandler

        analytical_handler = AnalyticalHandler()
        return analytical_handler.handle(query, reporting_tool, vector_service, params)
