# query_classifier.py
"""Query classification functionality for the routing system."""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List

from .extractor import EntityExtractor
from .types import QueryClassification


class QueryClassifier:
    """Classifies queries into different types based on patterns and content"""

    def __init__(self, entity_extractor: EntityExtractor):
        self.entity_extractor = entity_extractor
        self.classification_patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize regex patterns and keywords for query classification"""
        return {
            "report": [
                {
                    "pattern": r"\b(report|summary|overview|status)\b",
                    "weight": 0.8,
                    "keywords": [
                        "weekly",
                        "monthly",
                        "progress",
                        "achievements",
                        "blockers",
                    ],
                },
                {
                    "pattern": r"\b(generate|create|make)\s+(report|summary)\b",
                    "weight": 0.9,
                    "keywords": ["team", "project", "status"],
                },
            ],
            "factual": [
                {
                    "pattern": r"\b(who|what|when|where|how many|count|list)\b",
                    "weight": 0.7,
                    "keywords": [
                        "working",
                        "completed",
                        "involved",
                        "assigned",
                        "doing",
                    ],
                },
                {
                    "pattern": r"\b(status of|progress on|update on)\b",
                    "weight": 0.6,
                    "keywords": ["project", "task", "feature"],
                },
            ],
            "semantic": [
                {
                    "pattern": r"\b(similar|like|related|comparable|find.*about)\b",
                    "weight": 0.8,
                    "keywords": ["issues", "problems", "challenges", "solutions"],
                },
                {
                    "pattern": r"\b(theme|pattern|trend|common)\b",
                    "weight": 0.7,
                    "keywords": ["across", "team", "projects"],
                },
            ],
            "analytical": [
                {
                    "pattern": r"\b(why|analyze|compare|evaluate|impact|cause)\b",
                    "weight": 0.8,
                    "keywords": ["performance", "issues", "delay", "improvement"],
                },
                {
                    "pattern": r"\b(insight|conclusion|recommendation)\b",
                    "weight": 0.7,
                    "keywords": ["based", "analysis", "data"],
                },
            ],
            "comparison": [
                {
                    "pattern": r"\b(compare|vs|versus|against|between)\b",
                    "weight": 0.9,
                    "keywords": ["team", "performance", "progress", "velocity"],
                },
                {
                    "pattern": r"\b(difference|better|worse|more|less)\s+(than|compared)\b",
                    "weight": 0.8,
                    "keywords": ["last", "previous", "other"],
                },
            ],
        }

    def classify_query(self, query: str) -> QueryClassification:
        """Classify a query into different types based on patterns and content"""
        query_lower = query.lower()
        scores = {
            "report": 0.0,
            "factual": 0.0,
            "semantic": 0.0,
            "analytical": 0.0,
            "comparison": 0.0,
        }

        # Extract entities first
        entities = self.entity_extractor.extract_entities(query)

        # Pattern-based classification
        for query_type, patterns in self.classification_patterns.items():
            for pattern_info in patterns:
                if re.search(pattern_info["pattern"], query_lower):
                    scores[query_type] += pattern_info["weight"]

                # Keyword bonus
                keyword_matches = sum(
                    1 for keyword in pattern_info["keywords"] if keyword in query_lower
                )
                scores[query_type] += keyword_matches * 0.1

        # Apply special rules
        self._apply_special_rules(query_lower, scores, entities)

        # Determine best classification
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]

        # If confidence is too low, default to factual
        if confidence < 0.3:
            best_type = "factual"
            confidence = 0.5

        # Generate reasoning
        reasoning = self._generate_reasoning(query, best_type, scores)

        # Suggest parameters based on classification
        suggested_params = self._suggest_parameters(query, best_type)

        return QueryClassification(
            query_type=best_type,
            confidence=min(confidence, 1.0),
            reasoning=reasoning,
            suggested_params=suggested_params,
            entities=entities,
        )

    def _apply_special_rules(
        self, query_lower: str, scores: Dict[str, float], entities
    ) -> None:
        """Apply special classification rules"""

        # Date range detection for reports
        date_patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",  # YYYY-MM-DD
            r"\b(last|past|this)\s+(week|month|quarter)\b",
            r"\b(weekly|monthly|quarterly)\b",
            r"\bfrom\s+.*\bto\b.*\b",
        ]

        if any(re.search(pattern, query_lower) for pattern in date_patterns):
            scores["report"] += 0.5

        # Employee/role specific queries are often factual
        employee_pattern = self._get_dynamic_employee_pattern()
        role_pattern = self._get_dynamic_role_pattern()

        if re.search(employee_pattern, query_lower) or re.search(
            role_pattern, query_lower
        ):
            scores["factual"] += 0.3

        # Entity-based scoring boosts
        if entities.comparison_terms:
            scores["comparison"] += 0.7

        if entities.technologies:
            scores["semantic"] += 0.2  # Tech queries often need semantic search
            if len(entities.technologies) > 1:
                scores["comparison"] += 0.3  # Multiple techs suggest comparison

        if entities.focus_areas:
            if any(
                area in ["blocker", "issue", "problem", "bug"]
                for area in entities.focus_areas
            ):
                scores["analytical"] += 0.3  # Problem analysis
            if any(
                area in ["achievement", "success", "completion"]
                for area in entities.focus_areas
            ):
                scores["report"] += 0.2  # Achievements go in reports

        if entities.temporal_expressions:
            scores["report"] += 0.4  # Time-based queries often want reports
            if any("ago" in expr for expr in entities.temporal_expressions):
                scores["comparison"] += 0.2  # Comparing with past

        if entities.team_names and len(entities.team_names) > 1:
            scores["comparison"] += 0.4  # Multiple teams suggest comparison

    def _generate_reasoning(
        self, query: str, best_type: str, scores: Dict[str, float]
    ) -> str:
        """Generate reasoning for the classification decision"""
        reasons = []

        if best_type == "report":
            reasons.append("Contains report-generation keywords")
        elif best_type == "factual":
            reasons.append("Appears to be asking for specific facts or data")
        elif best_type == "semantic":
            reasons.append("Requires understanding of context and similarity")
        elif best_type == "analytical":
            reasons.append("Requires analysis and interpretation of data")
        elif best_type == "comparison":
            reasons.append("Requires comparison analysis")

        # Add confidence info
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
        if len(top_scores) > 1 and top_scores[0][1] - top_scores[1][1] < 0.2:
            reasons.append(
                f"Close decision between {top_scores[0][0]} and {top_scores[1][0]}"
            )

        return "; ".join(reasons)

    def _suggest_parameters(self, query: str, query_type: str) -> Dict[str, Any]:
        """Suggest parameters based on query content"""
        params = {}

        # Date range extraction
        self._extract_date_parameters(query, params)

        # Search parameters for semantic queries
        if query_type == "semantic":
            params["search_limit"] = 10
            params["use_vector_search"] = True

        # Employee/role filtering using dynamic patterns
        employee_pattern = self._get_dynamic_employee_pattern()
        role_pattern = self._get_dynamic_role_pattern()

        employee_match = re.search(employee_pattern, query.lower())
        if employee_match:
            params["employee_filter"] = employee_match.group(1).title()

        role_match = re.search(role_pattern, query.lower())
        if role_match:
            params["role_filter"] = role_match.group(1).title()

        return params

    def _extract_date_parameters(self, query: str, params: Dict[str, Any]) -> None:
        """Extract date-related parameters from query"""
        date_matches = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", query)
        query_lower = query.lower()

        if len(date_matches) >= 2:
            params["date_range"] = {"start": date_matches[0], "end": date_matches[-1]}
        elif "last week" in query_lower:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            params["date_range"] = {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            }
        elif any(
            word in query_lower
            for word in ["current", "recent", "latest", "now", "today"]
        ):
            # For "current/recent" queries, look at last 14 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)
            params["date_range"] = {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            }
        elif any(word in query_lower for word in ["this week", "weekly"]):
            # For "this week" queries
            end_date = datetime.now()
            # Go back to Monday of current week
            start_date = end_date - timedelta(days=end_date.weekday())
            params["date_range"] = {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            }
        elif any(word in query_lower for word in ["this month", "monthly"]):
            # For "this month" queries
            end_date = datetime.now()
            start_date = end_date.replace(day=1)  # First day of current month
            params["date_range"] = {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            }

        # Handle more complex temporal expressions
        self._handle_complex_temporal_expressions(query, query_lower, params)

    def _handle_complex_temporal_expressions(
        self, query: str, query_lower: str, params: Dict[str, Any]
    ) -> None:
        """Handle complex temporal expressions like 'N weeks ago', 'since January', etc."""

        # Handle "N weeks/months/days ago" patterns
        time_ago_match = re.search(r"\b(\d+)\s+(week|month|day)s?\s+ago\b", query_lower)
        if time_ago_match:
            number = int(time_ago_match.group(1))
            unit = time_ago_match.group(2)
            end_date = datetime.now()

            if unit == "day":
                start_date = end_date - timedelta(days=number)
            elif unit == "week":
                start_date = end_date - timedelta(weeks=number)
            elif unit == "month":
                # Approximate month calculation
                start_date = end_date - timedelta(days=number * 30)

            params["date_range"] = {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            }

        # Handle "since [month]" patterns
        since_month_match = re.search(
            r"\bsince\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            query_lower,
        )
        if since_month_match:
            month_name = since_month_match.group(1)
            month_map = {
                "january": 1,
                "february": 2,
                "march": 3,
                "april": 4,
                "may": 5,
                "june": 6,
                "july": 7,
                "august": 8,
                "september": 9,
                "october": 10,
                "november": 11,
                "december": 12,
            }

            current_date = datetime.now()
            month_num = month_map[month_name]

            # If the month is in the future, use previous year
            if month_num > current_date.month:
                year = current_date.year - 1
            else:
                year = current_date.year

            start_date = datetime(year, month_num, 1)
            params["date_range"] = {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": current_date.strftime("%Y-%m-%d"),
            }

        # Handle "Q1/Q2/Q3/Q4" patterns
        quarter_match = re.search(
            r"\bQ([1-4])\s*(?:20(\d{2}))?\b", query, re.IGNORECASE
        )
        if quarter_match:
            quarter = int(quarter_match.group(1))
            year = (
                int("20" + quarter_match.group(2))
                if quarter_match.group(2)
                else datetime.now().year
            )

            # Calculate quarter start and end dates
            quarter_start_month = (quarter - 1) * 3 + 1
            if quarter == 4:
                quarter_end_month = 12
                quarter_end_day = 31
            else:
                quarter_end_month = quarter * 3
                quarter_end_day = 30 if quarter_end_month in [4, 6, 9, 11] else 31
                if quarter_end_month == 2:
                    quarter_end_day = 29 if year % 4 == 0 else 28

            start_date = datetime(year, quarter_start_month, 1)
            end_date = datetime(year, quarter_end_month, quarter_end_day)

            params["date_range"] = {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            }

    def _get_dynamic_employee_pattern(self) -> str:
        """Generate regex pattern for employees from current data"""
        if not self.entity_extractor._cached_employees:
            return r"\b(team member|employee|person|individual)\b"

        # Escape special regex characters and create pattern
        escaped_employees = [
            re.escape(emp) for emp in self.entity_extractor._cached_employees
        ]
        return r"\b(" + "|".join(escaped_employees) + r")\b"

    def _get_dynamic_role_pattern(self) -> str:
        """Generate regex pattern for roles from current data"""
        if not self.entity_extractor._cached_roles:
            return r"\b(developer|engineer|designer|manager|analyst|tester)\b"

        # Escape special regex characters and create pattern
        escaped_roles = [
            re.escape(role) for role in self.entity_extractor._cached_roles
        ]
        return r"\b(" + "|".join(escaped_roles) + r")\b"
