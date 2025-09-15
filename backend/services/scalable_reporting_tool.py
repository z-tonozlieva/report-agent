# scalable_reporting_tool.py
"""Scalable reporting tool using repository pattern and database-aware operations."""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from backend.core import AggregatedData, LLMInterface, Update, settings
from backend.core.exceptions import QueryProcessingError
from backend.data import SQLAlchemyUpdateRepository


class ScalableReportingTool:
    """Scalable PM reporting tool using database-aware operations"""

    def __init__(self, llm: LLMInterface, repository: Optional[SQLAlchemyUpdateRepository] = None):
        self.llm = llm
        self.repository = repository or SQLAlchemyUpdateRepository()
        self.max_context_updates = settings.MAX_UPDATES_IN_MEMORY
        
    def add_update(self, update: Update) -> None:
        """Add a single update to the system"""
        self.repository.add(update)

    def add_updates(self, updates: List[Update]) -> None:
        """Add multiple updates to the system"""
        self.repository.add_many(updates)

    def clear_updates(self) -> None:
        """Clear all updates from the system"""
        self.repository.clear()

    def get_updates_by_date_range(
        self, start_date: datetime, end_date: datetime, limit: Optional[int] = None
    ) -> List[Update]:
        """Retrieve updates within a date range - SCALABLE"""
        updates = self.repository.get_by_date_range(start_date, end_date)
        
        if limit and len(updates) > limit:
            # Sort by date descending and take most recent
            updates = sorted(updates, key=lambda x: x.date, reverse=True)[:limit]
            
        return updates

    def get_updates_by_employee(self, employee_name: str) -> List[Update]:
        """Get updates for a specific employee"""
        return self.repository.get_by_employee(employee_name)

    def get_updates_by_role(self, role: str) -> List[Update]:
        """Get updates for a specific role"""
        return self.repository.get_by_role(role)
    
    def get_updates_by_department(self, department: str) -> List[Update]:
        """Get updates for a specific department"""
        # Use search functionality to find by department since repository doesn't have a direct method
        all_updates = self.repository.get_recent(limit=1000)  # Get reasonable sample
        return [u for u in all_updates if u.department and u.department.lower() == department.lower()]
    
    def get_updates_by_manager(self, manager: str) -> List[Update]:
        """Get updates for employees under a specific manager"""
        all_updates = self.repository.get_recent(limit=1000)  # Get reasonable sample
        return [u for u in all_updates if u.manager and u.manager.lower() == manager.lower()]

    def answer_contextual_question(
        self, 
        question: str, 
        date_range: Optional[Tuple[datetime, datetime]] = None,
        employee_filter: Optional[str] = None,
        role_filter: Optional[str] = None,
        max_updates: int = 50
    ) -> str:
        """Answer questions using FILTERED and LIMITED data - SCALABLE"""
        
        try:
            # Get filtered updates
            if date_range:
                updates = self.get_updates_by_date_range(
                    date_range[0], date_range[1], limit=max_updates
                )
            else:
                updates = self.repository.get_recent(limit=max_updates)
            
            # Apply additional filters
            if employee_filter:
                updates = [u for u in updates if u.employee.lower() == employee_filter.lower()]
            
            if role_filter:
                updates = [u for u in updates if u.role.lower() == role_filter.lower()]
            
            if not updates:
                return f"No updates found matching the specified criteria."
            
            # Limit context size
            updates = updates[:max_updates]
            
            # Create context from filtered updates
            context = "\n".join([
                f"{update.employee} ({update.role}" + 
                (f", {update.department}" if update.department else "") + 
                f", {update.date}): {update.update}"
                for update in updates
            ])
            
            # Check context size and truncate if needed
            if len(context) > settings.QUERY_TIMEOUT * 100:  # Rough estimate
                # Truncate to most recent updates
                context = "\n".join([
                    f"{update.employee} ({update.role}" + 
                    (f", {update.department}" if update.department else "") + 
                    f", {update.date}): {update.update}"
                    for update in updates[:20]  # Even more conservative
                ])
                context += f"\n\n[Note: Showing most recent 20 of {len(updates)} matching updates]"
            
            prompt = f"""
            Based on the following team updates, please answer this question:

            Question: {question}

            Team Updates ({len(updates)} updates):
            {context}

            Please provide a concise, data-driven answer based only on the provided updates.
            If the question cannot be answered from the available data, please state that clearly.
            """

            return self.llm.generate_response(prompt)
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to answer question: {str(e)}")

    def generate_smart_report(
        self, 
        date_range: Optional[Tuple[datetime, datetime]] = None,
        custom_prompt: Optional[str] = None,
        max_updates: int = None
    ) -> str:
        """Generate a report using SMART data selection - SCALABLE"""
        
        try:
            # Use memory-optimized limits
            if max_updates is None:
                max_updates = settings.MAX_REPORT_UPDATES if settings.LOW_MEMORY_MODE else 100
            
            # Get recent updates within date range
            if date_range:
                updates = self.get_updates_by_date_range(
                    date_range[0], date_range[1], limit=max_updates
                )
                date_context = f"from {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}"
            else:
                updates = self.repository.get_recent(limit=max_updates)
                date_context = f"recent {len(updates)} updates"
            
            if not updates:
                return "No updates available for the specified time period."
            
            # Aggregate data for better context
            aggregated = self.aggregate_updates(updates)
            
            # Use custom prompt or default
            if custom_prompt:
                prompt = f"{custom_prompt}\n\nBased on team updates {date_context}:"
            else:
                prompt = self._get_smart_report_prompt(aggregated, date_context)
            
            # Add structured context
            context = self._create_structured_context(aggregated)
            full_prompt = f"{prompt}\n\n{context}"
            
            return self.llm.generate_response(full_prompt)
            
        except Exception as e:
            raise QueryProcessingError(f"Failed to generate report: {str(e)}")

    def get_team_stats(self, date_range: Optional[Tuple[datetime, datetime]] = None) -> Dict:
        """Get team statistics - SCALABLE"""
        try:
            if date_range:
                updates = self.get_updates_by_date_range(date_range[0], date_range[1])
            else:
                updates = self.repository.get_recent(limit=1000)  # Get reasonable sample
            
            if not updates:
                return {"total_updates": 0, "employees": 0, "roles": 0}
            
            employees = set(u.employee for u in updates)
            roles = set(u.role for u in updates)
            departments = set(u.department for u in updates if u.department)
            
            role_distribution = defaultdict(int)
            employee_activity = defaultdict(int)
            department_distribution = defaultdict(int)
            
            for update in updates:
                role_distribution[update.role] += 1
                employee_activity[update.employee] += 1
                if update.department:
                    department_distribution[update.department] += 1
            
            return {
                "total_updates": len(updates),
                "employees": len(employees),
                "roles": len(roles),
                "departments": len(departments),
                "role_distribution": dict(role_distribution),
                "employee_activity": dict(employee_activity),
                "department_distribution": dict(department_distribution),
                "most_active_role": max(role_distribution.items(), key=lambda x: x[1])[0] if role_distribution else None,
                "most_active_employee": max(employee_activity.items(), key=lambda x: x[1])[0] if employee_activity else None,
                "most_active_department": max(department_distribution.items(), key=lambda x: x[1])[0] if department_distribution else None,
            }
        except Exception as e:
            raise QueryProcessingError(f"Failed to get team stats: {str(e)}")

    def aggregate_updates(self, updates: Optional[List[Update]] = None) -> AggregatedData:
        """Aggregate updates for report generation - memory efficient"""
        if updates is None:
            updates = self.repository.get_recent(limit=self.max_context_updates)

        # Group by role and extract key information
        role_updates = defaultdict(list)
        projects = set()
        achievements = []
        blockers = []

        for update in updates:
            role_updates[update.role].append(update)

            # Extract projects (simple keyword matching)
            if any(word in update.update.lower() for word in ['project', 'app', 'system', 'platform']):
                # Extract potential project names
                words = update.update.split()
                for i, word in enumerate(words):
                    if word.lower() in ['project', 'app', 'system', 'platform'] and i > 0:
                        projects.add(words[i-1].capitalize())

            # Extract achievements and blockers
            if any(word in update.update.lower() for word in ['completed', 'finished', 'delivered', 'launched', 'released']):
                achievements.append(f"{update.employee}: {update.update}")

            if any(word in update.update.lower() for word in ['blocked', 'blocker', 'issue', 'problem', 'stuck']):
                blockers.append(f"{update.employee}: {update.update}")

        return AggregatedData(
            role_updates=dict(role_updates),
            projects=list(projects),
            achievements=achievements[:10],  # Limit to prevent context explosion
            blockers=blockers[:10],  # Limit to prevent context explosion
            total_updates=len(updates)
        )

    def _get_smart_report_prompt(self, aggregated: AggregatedData, date_context: str) -> str:
        """Generate smart report prompt based on aggregated data"""
        # aggregated parameter reserved for future enhanced prompting
        return f"""
        Generate a comprehensive team status report based on the following aggregated team data {date_context}.
        
        Focus on:
        - Overall team progress and achievements
        - Current blockers and challenges  
        - Project status and key developments
        - Team productivity and coordination
        - Recommendations for leadership
        
        Structure the report with clear sections and actionable insights.
        """

    def _create_structured_context(self, aggregated: AggregatedData) -> str:
        """Create structured context from aggregated data"""
        context_parts = [
            f"=== TEAM OVERVIEW ===",
            f"Total Updates: {aggregated.total_updates}",
            f"Active Roles: {len(aggregated.role_updates)}",
            f"Projects Mentioned: {', '.join(aggregated.projects) if aggregated.projects else 'None identified'}",
        ]
        
        if aggregated.achievements:
            context_parts.extend([
                f"\n=== KEY ACHIEVEMENTS ===",
                *aggregated.achievements[:5]  # Top 5 achievements
            ])
        
        if aggregated.blockers:
            context_parts.extend([
                f"\n=== CURRENT BLOCKERS ===", 
                *aggregated.blockers[:5]  # Top 5 blockers
            ])
        
        context_parts.append(f"\n=== DETAILED UPDATES BY ROLE ===")
        for role, updates in aggregated.role_updates.items():
            context_parts.append(f"\n--- {role} ({len(updates)} updates) ---")
            for update in updates[:3]:  # Max 3 updates per role for context
                dept_info = f", {update.department}" if update.department else ""
                context_parts.append(f"• {update.employee}{dept_info} ({update.date}): {update.update}")
            if len(updates) > 3:
                context_parts.append(f"... and {len(updates) - 3} more updates")
        
        return "\n".join(context_parts)

