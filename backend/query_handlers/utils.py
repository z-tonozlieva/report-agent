# query_utils.py
"""Utility functions for query routing and processing."""

import re
from typing import Set


class QueryPatternUtils:
    """Utility class for query pattern matching and extraction"""

    @staticmethod
    def get_dynamic_employee_pattern(cached_employees: Set[str]) -> str:
        """Generate regex pattern for employees from current data"""
        if not cached_employees:
            return r"\b(team member|employee|person|individual)\b"

        # Escape special regex characters and create pattern
        escaped_employees = [re.escape(emp) for emp in cached_employees]
        return r"\b(" + "|".join(escaped_employees) + r")\b"

    @staticmethod
    def get_dynamic_role_pattern(cached_roles: Set[str]) -> str:
        """Generate regex pattern for roles from current data"""
        if not cached_roles:
            return r"\b(developer|engineer|designer|manager|analyst|tester)\b"

        # Escape special regex characters and create pattern
        escaped_roles = [re.escape(role) for role in cached_roles]
        return r"\b(" + "|".join(escaped_roles) + r")\b"
