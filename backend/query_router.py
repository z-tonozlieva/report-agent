# query_router.py
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from .interfaces import LLMInterface
from .entity_config import get_entity_config

logger = logging.getLogger(__name__)


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


class SmartQueryRouter:
    """Intelligent query router that directs queries to the most appropriate backend"""

    def __init__(self, llm: Optional[LLMInterface] = None):
        self.llm = llm
        self.classification_patterns = self._initialize_patterns()
        self.entity_config = get_entity_config()
        self._cached_employees = set()
        self._cached_roles = set()
        self._cached_projects = set()
        self._cached_technologies = set()

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

    def extract_entities(self, query: str) -> EntityExtraction:
        """Extract entities from a query using configurable patterns"""
        query_lower = query.lower()
        
        # Use configuration-based entity extraction
        projects = self.entity_config.find_matching_projects(query)
        technologies = self.entity_config.find_matching_technologies(query)
        team_names = self.entity_config.find_matching_teams(query)
        focus_areas = self.entity_config.find_matching_focus_areas(query)
        
        # Extract comparison terms (keep existing logic)
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
        
        # Extract temporal expressions (keep existing logic)
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
        
        # Add cached entities from current data
        if hasattr(self, '_cached_projects'):
            for project in self._cached_projects:
                if project.lower() in query_lower:
                    projects.add(project)
        
        if hasattr(self, '_cached_technologies'):
            for tech in self._cached_technologies:
                if tech.lower() in query_lower:
                    technologies.add(tech)
        
        # Check if any employees have team roles
        for employee in self._cached_employees:
            # Find if this employee's role matches any team
            employee_updates = [u for u in getattr(self, '_current_updates', []) if u.employee.lower() == employee]
            if employee_updates:
                role = employee_updates[0].role
                team = self.entity_config.get_team_by_role(role)
                if team:
                    team_names.add(team)
        
        return EntityExtraction(
            projects=projects,
            technologies=technologies,
            team_names=team_names,
            focus_areas=focus_areas,
            comparison_terms=comparison_terms,
            temporal_expressions=temporal_expressions,
        )

    def classify_query(self, query: str) -> QueryClassification:
        """Classify a query into different types based on patterns and content"""
        query_lower = query.lower()
        scores = {"report": 0.0, "factual": 0.0, "semantic": 0.0, "analytical": 0.0, "comparison": 0.0}
        
        # Extract entities first
        entities = self.extract_entities(query)

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

        # Special rules

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
            if any(area in ["blocker", "issue", "problem", "bug"] for area in entities.focus_areas):
                scores["analytical"] += 0.3  # Problem analysis
            if any(area in ["achievement", "success", "completion"] for area in entities.focus_areas):
                scores["report"] += 0.2  # Achievements go in reports
        
        if entities.temporal_expressions:
            scores["report"] += 0.4  # Time-based queries often want reports
            if any("ago" in expr for expr in entities.temporal_expressions):
                scores["comparison"] += 0.2  # Comparing with past
        
        if entities.team_names and len(entities.team_names) > 1:
            scores["comparison"] += 0.4  # Multiple teams suggest comparison

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
        
        # Advanced temporal expressions
        # Handle "N weeks/months/days ago" patterns
        time_ago_match = re.search(r'\b(\d+)\s+(week|month|day)s?\s+ago\b', query_lower)
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
        since_month_match = re.search(r'\bsince\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b', query_lower)
        if since_month_match:
            month_name = since_month_match.group(1)
            month_map = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12
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
        quarter_match = re.search(r'\bQ([1-4])\s*(?:20(\d{2}))?\b', query, re.IGNORECASE)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            year = int("20" + quarter_match.group(2)) if quarter_match.group(2) else datetime.now().year
            
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

        # No longer using report types - simplified approach

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
            classification = self.classify_query(query)

        logger.info(
            f"Query classified as '{classification.query_type}' with confidence {classification.confidence:.2f}"
        )

        try:
            if classification.query_type == "report":
                answer, additional_info = self._handle_report_query(
                    query, reporting_tool, classification.suggested_params
                )
            elif classification.query_type == "semantic":
                answer, additional_info = self._handle_semantic_query(
                    query,
                    vector_service,
                    reporting_tool,
                    classification.suggested_params,
                )
            elif classification.query_type == "analytical":
                answer, additional_info = self._handle_analytical_query(
                    query,
                    reporting_tool,
                    vector_service,
                    classification.suggested_params,
                )
            elif classification.query_type == "comparison":
                answer, additional_info = self._handle_comparison_query(
                    query,
                    reporting_tool,
                    vector_service,
                    classification.suggested_params,
                    classification.entities,
                )
            else:  # factual
                answer, additional_info = self._handle_factual_query(
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
            # Fallback to simple factual query
            try:
                answer = reporting_tool.answer_factual_question_from_all_data(query)
                return RouterResponse(
                    answer=f"(Fallback) {answer}",
                    method_used="factual_fallback",
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

    def _handle_report_query(
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

        report = reporting_tool.generate_report(
            date_range=date_range_tuple, custom_prompt=custom_prompt
        )

        additional_info = {
            "date_range": date_range,
            "method": "report_generation",
            "custom_prompt_used": True,
        }

        return report, additional_info

    def _handle_semantic_query(
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

    def _handle_analytical_query(
        self, query: str, reporting_tool, vector_service, params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle analytical queries that require deeper reasoning"""

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

            answer = reporting_tool.answer_factual_question_from_all_data(
                enhanced_query
            )
        else:
            answer = reporting_tool.answer_factual_question_from_all_data(query)

        additional_info = {
            "method": "analytical_hybrid",
            "semantic_context_used": len(semantic_results) > 0,
            "context_updates": len(semantic_results),
        }

        return answer, additional_info

    def _handle_factual_query(
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

    def _handle_comparison_query(
        self,
        query: str,
        reporting_tool,
        vector_service,
        params: Dict[str, Any],
        entities: Optional[EntityExtraction] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle comparison queries that compare teams, technologies, time periods, etc."""
        
        if not entities:
            # Fallback to analytical query
            return self._handle_analytical_query(query, reporting_tool, vector_service, params)
        
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
                    update for update in reporting_tool.updates
                    if team.lower() in update.update.lower() or team.lower() in update.role.lower()
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
                    update for update in reporting_tool.updates
                    if tech.lower() in update.update.lower()
                ]
                tech_data[tech] = tech_updates
            
            comparison_filters.append({"type": "technology", "data": tech_data})
        
        # Time period comparison (if temporal expressions suggest comparison)
        if entities.temporal_expressions and any("ago" in expr for expr in entities.temporal_expressions):
            comparison_context.append("Comparing time periods")
        
        # Role comparison
        if entities.comparison_terms:
            terms = list(entities.comparison_terms)
            # Check if comparison terms match roles/employees
            role_matches = []
            employee_matches = []
            
            for term in terms:
                if term.lower() in self._cached_roles:
                    role_matches.append(term)
                elif term.lower() in self._cached_employees:
                    employee_matches.append(term)
            
            if role_matches:
                comparison_context.append(f"Comparing roles: {', '.join(role_matches)}")
            if employee_matches:
                comparison_context.append(f"Comparing employees: {', '.join(employee_matches)}")
        
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
            all_updates_context = "\n".join([
                f"{update.employee} ({update.role}, {update.date}): {update.update}"
                for update in reporting_tool.updates[-20:]  # Last 20 updates
            ])
            
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
                "technologies": list(entities.technologies) if entities.technologies else [],
                "comparison_terms": list(entities.comparison_terms) if entities.comparison_terms else [],
            },
            "semantic_context_used": len(semantic_results) > 0,
            "comparison_filters": len(comparison_filters),
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

    def update_dynamic_patterns(self, reporting_tool) -> None:
        """Update employee, role, project and technology patterns from current data"""
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
                        r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+(?:project|feature|system|platform|dashboard|portal|service|module)\b',
                        update.update,
                        re.IGNORECASE
                    )
                    self._cached_projects.update(p.strip() for p in project_indicators if p.strip())
                    
                    # Use entity config to find technologies mentioned in updates
                    update_techs = self.entity_config.find_matching_technologies(update.update)
                    self._cached_technologies.update(update_techs)

    def _get_dynamic_employee_pattern(self) -> str:
        """Generate regex pattern for employees from current data"""
        if not self._cached_employees:
            return r"\b(team member|employee|person|individual)\b"

        # Escape special regex characters and create pattern
        escaped_employees = [re.escape(emp) for emp in self._cached_employees]
        return r"\b(" + "|".join(escaped_employees) + r")\b"

    def _get_dynamic_role_pattern(self) -> str:
        """Generate regex pattern for roles from current data"""
        if not self._cached_roles:
            return r"\b(developer|engineer|designer|manager|analyst|tester)\b"

        # Escape special regex characters and create pattern
        escaped_roles = [re.escape(role) for role in self._cached_roles]
        return r"\b(" + "|".join(escaped_roles) + r")\b"
