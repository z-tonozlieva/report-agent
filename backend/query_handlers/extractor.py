# entity_extractor.py
"""Entity extraction functionality for query routing."""

import re
from typing import Set

from .entity_config import get_entity_config
from .types import EntityExtraction


class EntityExtractor:
    """Extracts entities from queries using configurable patterns"""

    def __init__(self):
        self.entity_config = get_entity_config()
        self._cached_employees = set()
        self._cached_roles = set()
        self._cached_projects = set()
        self._cached_technologies = set()
        self._current_updates = []

    def extract_entities(self, query: str) -> EntityExtraction:
        """Extract entities from a query using configurable patterns"""
        query_lower = query.lower()

        # Use configuration-based entity extraction
        projects = self.entity_config.find_matching_projects(query)
        technologies = self.entity_config.find_matching_technologies(query)
        team_names = self.entity_config.find_matching_teams(query)
        focus_areas = self.entity_config.find_matching_focus_areas(query)

        # Extract comparison terms
        comparison_terms = self._extract_comparison_terms(query)

        # Extract temporal expressions
        temporal_expressions = self._extract_temporal_expressions(query_lower)

        # Add cached entities from current data
        projects.update(self._find_cached_projects(query_lower))
        technologies.update(self._find_cached_technologies(query_lower))

        # Check if any employees have team roles
        team_names.update(self._find_team_names_from_employees(query_lower))

        return EntityExtraction(
            projects=projects,
            technologies=technologies,
            team_names=team_names,
            focus_areas=focus_areas,
            comparison_terms=comparison_terms,
            temporal_expressions=temporal_expressions,
        )

    def _extract_comparison_terms(self, query: str) -> Set[str]:
        """Extract comparison terms from query"""
        comparison_terms = set()
        comparison_patterns = [
            r"\bcompare\s+(\w+(?:\s+\w+)*)\s+(?:vs|versus|against|with|to)\s+(\w+(?:\s+\w+)*)\b",
            r"\b(\w+(?:\s+\w+)*)\s+vs\s+(\w+(?:\s+\w+)*)\b",
            r"\bbetween\s+(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)\b",
        ]

        for pattern in comparison_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    comparison_terms.update(match)

        return comparison_terms

    def _extract_temporal_expressions(self, query_lower: str) -> Set[str]:
        """Extract temporal expressions from query"""
        temporal_expressions = set()
        temporal_patterns = [
            r"\b(\d+)\s+(sprint|week|month|quarter|day)s?\s+ago\b",
            r"\b(last|previous|current|this|next)\s+(sprint|week|month|quarter|year)\b",
            r"\bsince\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            r"\bbefore\s+(christmas|holidays|vacation|release)\b",
            r"\bQ[1-4]\s*(?:20\d{2})?\b",
        ]

        for pattern in temporal_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        temporal_expressions.add(" ".join(match))
                    else:
                        temporal_expressions.add(match)

        return temporal_expressions

    def _find_cached_projects(self, query_lower: str) -> Set[str]:
        """Find cached projects mentioned in the query"""
        projects = set()
        if hasattr(self, "_cached_projects"):
            for project in self._cached_projects:
                if project.lower() in query_lower:
                    projects.add(project)
        return projects

    def _find_cached_technologies(self, query_lower: str) -> Set[str]:
        """Find cached technologies mentioned in the query"""
        technologies = set()
        if hasattr(self, "_cached_technologies"):
            for tech in self._cached_technologies:
                if tech.lower() in query_lower:
                    technologies.add(tech)
        return technologies

    def _find_team_names_from_employees(self, query_lower: str) -> Set[str]:
        """Find team names based on employee roles"""
        team_names = set()

        for employee in self._cached_employees:
            # Find if this employee's role matches any team
            employee_updates = [
                u for u in self._current_updates if u.employee.lower() == employee
            ]
            if employee_updates:
                role = employee_updates[0].role
                team = self.entity_config.get_team_by_role(role)
                if team:
                    team_names.add(team)

        return team_names

    def update_cache(self, reporting_tool) -> None:
        """Update cached entities from current reporting tool data"""
        if hasattr(reporting_tool, "updates") and reporting_tool.updates:
            # Store current updates for team role matching
            self._current_updates = reporting_tool.updates

            self._cached_employees = {
                update.employee.lower().strip()
                for update in reporting_tool.updates
                if update.employee and update.employee.strip()
            }
            self._cached_roles = {
                update.role.lower().strip()
                for update in reporting_tool.updates
                if update.role and update.role.strip()
            }

            # Extract projects and technologies from update content
            self._cached_projects = set()
            self._cached_technologies = set()

            for update in reporting_tool.updates:
                if update.update:
                    # Extract common project terms using existing patterns
                    project_indicators = re.findall(
                        r"\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+(?:project|feature|system|platform|dashboard|portal|service|module)\b",
                        update.update,
                        re.IGNORECASE,
                    )
                    self._cached_projects.update(
                        p.strip() for p in project_indicators if p.strip()
                    )

                    # Use entity config to find technologies mentioned in updates
                    update_techs = self.entity_config.find_matching_technologies(
                        update.update
                    )
                    self._cached_technologies.update(update_techs)
