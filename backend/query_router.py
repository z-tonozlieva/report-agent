# query_router.py
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .interfaces import LLMInterface

logger = logging.getLogger(__name__)


@dataclass
class QueryClassification:
    """Classification result for a query"""
    query_type: str  # 'report', 'factual', 'semantic', 'analytical'
    confidence: float
    reasoning: str
    suggested_params: Dict[str, Any]


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
    
    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize regex patterns and keywords for query classification"""
        return {
            'report': [
                {
                    'pattern': r'\b(report|summary|overview|status)\b',
                    'weight': 0.8,
                    'keywords': ['weekly', 'monthly', 'progress', 'achievements', 'blockers']
                },
                {
                    'pattern': r'\b(generate|create|make)\s+(report|summary)\b',
                    'weight': 0.9,
                    'keywords': ['team', 'project', 'status']
                }
            ],
            'factual': [
                {
                    'pattern': r'\b(who|what|when|where|how many|count|list)\b',
                    'weight': 0.7,
                    'keywords': ['working', 'completed', 'involved', 'assigned', 'doing']
                },
                {
                    'pattern': r'\b(status of|progress on|update on)\b',
                    'weight': 0.6,
                    'keywords': ['project', 'task', 'feature']
                }
            ],
            'semantic': [
                {
                    'pattern': r'\b(similar|like|related|comparable|find.*about)\b',
                    'weight': 0.8,
                    'keywords': ['issues', 'problems', 'challenges', 'solutions']
                },
                {
                    'pattern': r'\b(theme|pattern|trend|common)\b',
                    'weight': 0.7,
                    'keywords': ['across', 'team', 'projects']
                }
            ],
            'analytical': [
                {
                    'pattern': r'\b(why|analyze|compare|evaluate|impact|cause)\b',
                    'weight': 0.8,
                    'keywords': ['performance', 'issues', 'delay', 'improvement']
                },
                {
                    'pattern': r'\b(insight|conclusion|recommendation)\b',
                    'weight': 0.7,
                    'keywords': ['based', 'analysis', 'data']
                }
            ]
        }
    
    def classify_query(self, query: str) -> QueryClassification:
        """Classify a query into different types based on patterns and content"""
        query_lower = query.lower()
        scores = {'report': 0.0, 'factual': 0.0, 'semantic': 0.0, 'analytical': 0.0}
        
        # Pattern-based classification
        for query_type, patterns in self.classification_patterns.items():
            for pattern_info in patterns:
                if re.search(pattern_info['pattern'], query_lower):
                    scores[query_type] += pattern_info['weight']
                
                # Keyword bonus
                keyword_matches = sum(1 for keyword in pattern_info['keywords'] 
                                    if keyword in query_lower)
                scores[query_type] += keyword_matches * 0.1
        
        # Special rules
        
        # Date range detection for reports
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b(last|past|this)\s+(week|month|quarter)\b',
            r'\b(weekly|monthly|quarterly)\b',
            r'\bfrom\s+.*\bto\b.*\b'
        ]
        
        if any(re.search(pattern, query_lower) for pattern in date_patterns):
            scores['report'] += 0.5
        
        # Employee/role specific queries are often factual
        if re.search(r'\b(john|sarah|alice|bob|team|developer|designer|manager)\b', query_lower):
            scores['factual'] += 0.3
        
        # Determine best classification
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        # If confidence is too low, default to factual
        if confidence < 0.3:
            best_type = 'factual'
            confidence = 0.5
        
        # Generate reasoning
        reasoning = self._generate_reasoning(query, best_type, scores)
        
        # Suggest parameters based on classification
        suggested_params = self._suggest_parameters(query, best_type)
        
        return QueryClassification(
            query_type=best_type,
            confidence=min(confidence, 1.0),
            reasoning=reasoning,
            suggested_params=suggested_params
        )
    
    def _generate_reasoning(self, query: str, best_type: str, scores: Dict[str, float]) -> str:
        """Generate reasoning for the classification decision"""
        reasons = []
        
        if best_type == 'report':
            reasons.append("Contains report-generation keywords")
        elif best_type == 'factual':
            reasons.append("Appears to be asking for specific facts or data")
        elif best_type == 'semantic':
            reasons.append("Requires understanding of context and similarity")
        elif best_type == 'analytical':
            reasons.append("Requires analysis and interpretation of data")
        
        # Add confidence info
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
        if len(top_scores) > 1 and top_scores[0][1] - top_scores[1][1] < 0.2:
            reasons.append(f"Close decision between {top_scores[0][0]} and {top_scores[1][0]}")
        
        return "; ".join(reasons)
    
    def _suggest_parameters(self, query: str, query_type: str) -> Dict[str, Any]:
        """Suggest parameters based on query content"""
        params = {}
        
        # Date range extraction
        date_matches = re.findall(r'\b(\d{4}-\d{2}-\d{2})\b', query)
        if len(date_matches) >= 2:
            params['date_range'] = {'start': date_matches[0], 'end': date_matches[-1]}
        elif 'last week' in query.lower():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            params['date_range'] = {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }
        
        # Report type detection
        if query_type == 'report':
            if 'weekly' in query.lower():
                params['report_type'] = 'weekly'
            elif 'blocker' in query.lower() or 'challenge' in query.lower():
                params['report_type'] = 'blockers'
            elif 'achievement' in query.lower() or 'accomplish' in query.lower():
                params['report_type'] = 'achievements'
            else:
                params['report_type'] = 'weekly'
        
        # Search parameters for semantic queries
        if query_type == 'semantic':
            params['search_limit'] = 10
            params['use_vector_search'] = True
        
        # Employee/role filtering
        employee_pattern = r'\b(john|sarah|alice|bob|charlie|diana|eve|frank)\b'
        role_pattern = r'\b(developer|designer|manager|tester|analyst)\b'
        
        employee_match = re.search(employee_pattern, query.lower())
        if employee_match:
            params['employee_filter'] = employee_match.group(1).title()
        
        role_match = re.search(role_pattern, query.lower())
        if role_match:
            params['role_filter'] = role_match.group(1).title()
        
        return params
    
    def route_query(
        self, 
        query: str, 
        reporting_tool, 
        vector_service, 
        db_session=None,
        override_method: Optional[str] = None
    ) -> RouterResponse:
        """Route a query to the most appropriate backend and return response"""
        
        # Classify query unless method is overridden
        if override_method:
            classification = QueryClassification(
                query_type=override_method,
                confidence=1.0,
                reasoning=f"Method overridden to {override_method}",
                suggested_params={}
            )
        else:
            classification = self.classify_query(query)
        
        logger.info(f"Query classified as '{classification.query_type}' with confidence {classification.confidence:.2f}")
        
        try:
            if classification.query_type == 'report':
                answer, additional_info = self._handle_report_query(
                    query, reporting_tool, classification.suggested_params
                )
            elif classification.query_type == 'semantic':
                answer, additional_info = self._handle_semantic_query(
                    query, vector_service, reporting_tool, classification.suggested_params
                )
            elif classification.query_type == 'analytical':
                answer, additional_info = self._handle_analytical_query(
                    query, reporting_tool, vector_service, classification.suggested_params
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
                    'classification': classification,
                    'routing_info': additional_info
                }
            )
            
        except Exception as e:
            logger.error(f"Error routing query: {str(e)}")
            # Fallback to simple factual query
            try:
                answer = reporting_tool.answer_stakeholder_question(query)
                return RouterResponse(
                    answer=f"(Fallback) {answer}",
                    method_used='factual_fallback',
                    confidence=0.5,
                    additional_info={'error': str(e)}
                )
            except Exception as fallback_error:
                return RouterResponse(
                    answer=f"Unable to process query: {str(fallback_error)}",
                    method_used='error',
                    confidence=0.0,
                    additional_info={'error': str(fallback_error)}
                )
    
    def _handle_report_query(
        self, 
        query: str, 
        reporting_tool, 
        params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle report generation queries"""
        
        report_type = params.get('report_type', 'weekly')
        date_range = params.get('date_range')
        
        if date_range:
            start_date = datetime.strptime(date_range['start'], '%Y-%m-%d')
            end_date = datetime.strptime(date_range['end'], '%Y-%m-%d')
            report = reporting_tool.generate_report(
                report_type=report_type,
                date_range=(start_date, end_date),
                custom_prompt=None
            )
        else:
            # Use custom prompt for report generation
            custom_prompt = f"""
            Generate a {report_type} report based on the following request:
            
            "{query}"
            
            Structure the response as requested, focusing on the specific aspects mentioned in the query.
            """
            
            report = reporting_tool.generate_report(
                report_type=report_type,
                custom_prompt=custom_prompt
            )
        
        additional_info = {
            'report_type': report_type,
            'date_range': date_range,
            'method': 'report_generation'
        }
        
        return report, additional_info
    
    def _handle_semantic_query(
        self, 
        query: str, 
        vector_service, 
        reporting_tool, 
        params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle semantic search queries"""
        
        search_limit = params.get('search_limit', 10)
        
        # Build search filters
        filters = {}
        if 'employee_filter' in params:
            filters['employee'] = params['employee_filter']
        if 'role_filter' in params:
            filters['role'] = params['role_filter']
        if 'date_range' in params:
            filters['date_from'] = params['date_range']['start']
            filters['date_to'] = params['date_range']['end']
        
        # Perform semantic search
        search_results = vector_service.semantic_search(
            query, 
            limit=search_limit, 
            filters=filters if filters else None
        )
        
        if not search_results:
            # Fallback to traditional Q&A
            answer = reporting_tool.answer_stakeholder_question(query)
            additional_info = {
                'method': 'semantic_fallback_to_traditional',
                'search_results': 0
            }
            return f"(No semantic matches found) {answer}", additional_info
        
        # Format semantic results into a comprehensive answer
        answer = self._format_semantic_results(query, search_results)
        
        additional_info = {
            'method': 'semantic_search',
            'search_results': len(search_results),
            'top_matches': [
                {
                    'employee': result['employee'],
                    'relevance': round(result['similarity_score'], 3),
                    'snippet': result['document'][:100] + '...' if len(result['document']) > 100 else result['document']
                }
                for result in search_results[:3]
            ]
        }
        
        return answer, additional_info
    
    def _handle_analytical_query(
        self, 
        query: str, 
        reporting_tool, 
        vector_service, 
        params: Dict[str, Any]
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
            
            answer = reporting_tool.answer_stakeholder_question(enhanced_query)
        else:
            answer = reporting_tool.answer_stakeholder_question(query)
        
        additional_info = {
            'method': 'analytical_hybrid',
            'semantic_context_used': len(semantic_results) > 0,
            'context_updates': len(semantic_results)
        }
        
        return answer, additional_info
    
    def _handle_factual_query(
        self, 
        query: str, 
        reporting_tool, 
        params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle straightforward factual queries"""
        
        answer = reporting_tool.answer_stakeholder_question(query)
        
        additional_info = {
            'method': 'traditional_qa',
            'filters_applied': bool(params.get('employee_filter') or params.get('role_filter'))
        }
        
        return answer, additional_info
    
    def _format_semantic_results(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Format semantic search results into a comprehensive answer"""
        
        if not results:
            return "No relevant information found."
        
        # Group results by relevance
        high_relevance = [r for r in results if r['similarity_score'] > 0.8]
        medium_relevance = [r for r in results if 0.6 <= r['similarity_score'] <= 0.8]
        
        answer_parts = [f"Based on semantic analysis of team updates, here's what I found regarding: '{query}'"]
        
        if high_relevance:
            answer_parts.append("\n**Most Relevant Updates:**")
            for result in high_relevance[:3]:
                answer_parts.append(
                    f"• {result['employee']} ({result['role']}, {result['date']}): "
                    f"{result['document']} (Relevance: {result['similarity_score']:.2f})"
                )
        
        if medium_relevance:
            answer_parts.append("\n**Additional Related Updates:**")
            for result in medium_relevance[:2]:
                answer_parts.append(
                    f"• {result['employee']} ({result['role']}, {result['date']}): "
                    f"{result['document'][:150]}... (Relevance: {result['similarity_score']:.2f})"
                )
        
        # Add summary if we have enough results
        if len(results) >= 3:
            employees = list(set(r['employee'] for r in results))
            roles = list(set(r['role'] for r in results))
            answer_parts.append(
                f"\n**Summary:** Found {len(results)} relevant updates from "
                f"{len(employees)} team members across {len(roles)} roles."
            )
        
        return "\n".join(answer_parts)
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Provide query suggestions based on partial input"""
        suggestions = []
        
        partial_lower = partial_query.lower()
        
        if 'report' in partial_lower or 'summary' in partial_lower:
            suggestions.extend([
                "Generate a weekly report for the last 7 days",
                "Create a summary of blockers and challenges",
                "Show me a status report for the development team"
            ])
        
        if any(word in partial_lower for word in ['who', 'what', 'when']):
            suggestions.extend([
                "Who is working on the payment integration?",
                "What are the current blockers for the project?",
                "When was the last update from Sarah?"
            ])
        
        if any(word in partial_lower for word in ['similar', 'like', 'find']):
            suggestions.extend([
                "Find updates similar to database issues",
                "Show me updates like authentication problems",
                "What are common themes across the team?"
            ])
        
        return suggestions[:5]  # Return top 5 suggestions