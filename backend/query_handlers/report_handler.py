# report_handler.py
"""Handler for report generation queries."""

from datetime import datetime
from typing import Any, Dict, Tuple


class ReportHandler:
    """Handles report generation queries"""

    def handle(
        self, query: str, reporting_tool, params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle report generation queries"""

        date_range = params.get("date_range")
        date_range_tuple = None

        if date_range:
            start_date = datetime.strptime(date_range["start"], "%Y-%m-%d")
            end_date = datetime.strptime(date_range["end"], "%Y-%m-%d")
            date_range_tuple = (start_date, end_date)

        # Use query as custom prompt for report generation
        custom_prompt = f"""
        Generate a team status report based on the following request:

        "{query}"

        Structure the response as requested, focusing on the specific aspects mentioned in the query.
        """

        # Use scalable report generation
        report = reporting_tool.generate_smart_report(
            date_range=date_range_tuple, 
            custom_prompt=custom_prompt,
            max_updates=150  # Reasonable limit for reports
        )
        method_used = "smart_report_generation"

        additional_info = {
            "date_range": date_range,
            "method": method_used,
            "custom_prompt_used": True,
            "max_updates_limit": 150,
        }

        return report, additional_info
