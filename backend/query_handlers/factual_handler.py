# factual_handler.py
"""Handler for straightforward factual queries."""

from datetime import datetime
from typing import Any, Dict, Tuple


class FactualHandler:
    """Handles straightforward factual queries with optional date filtering"""

    def handle(
        self, query: str, reporting_tool, params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle straightforward factual queries with optional date filtering"""

        date_range = params.get("date_range")

        if date_range:
            # Use date-filtered factual query
            start_date = datetime.strptime(date_range["start"], "%Y-%m-%d")
            end_date = datetime.strptime(date_range["end"], "%Y-%m-%d")

            # Get updates within date range
            filtered_updates = reporting_tool.get_updates_by_date_range(
                start_date, end_date
            )

            if not filtered_updates:
                answer = f"No updates found within the specified timeframe ({date_range['start']} to {date_range['end']}) to answer the question."
            else:
                # Create context from filtered updates
                context = "\n".join(
                    [
                        f"{update.employee} ({update.role}, {update.date}): {update.update}"
                        for update in filtered_updates
                    ]
                )

                prompt = f"""
                Based on the following team updates from {date_range["start"]} to {date_range["end"]}, please answer this question:

                Question: {query}

                Team Updates:
                {context}

                Please provide a concise, data-driven answer based only on the updates within this timeframe.
                """

                answer = reporting_tool.llm.generate_response(prompt)

            additional_info = {
                "method": "factual_date_filtered",
                "date_range": date_range,
                "updates_count": len(filtered_updates) if filtered_updates else 0,
                "filters_applied": True,
            }
        else:
            # Use all data (original behavior)
            answer = reporting_tool.answer_factual_question_from_all_data(query)

            additional_info = {
                "method": "factual_all_data",
                "warning": "Uses ALL updates in database",
                "filters_applied": bool(
                    params.get("employee_filter") or params.get("role_filter")
                ),
            }

        return answer, additional_info
