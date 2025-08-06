# reporting_tool.py
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .models import Update, AggregatedData
from .mock_llm import LLMInterface
from .data_loader import DataLoader

class PMReportingTool:
    """Main class for the PM reporting tool"""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.updates: List[Update] = []
        
    def add_update(self, update: Update) -> None:
        """Add a single update to the system"""
        self.updates.append(update)
    
    def add_updates(self, updates: List[Update]) -> None:
        """Add multiple updates to the system"""
        self.updates.extend(updates)
    
    def load_mock_data(self) -> None:
        """Load mock employee updates"""
        mock_updates = DataLoader.get_mock_updates()
        self.add_updates(mock_updates)
    
    def load_additional_mock_data(self) -> None:
        """Load additional mock data for testing"""
        additional_updates = DataLoader.get_additional_mock_updates()
        self.add_updates(additional_updates)
    
    def load_blocker_data(self) -> None:
        """Load mock data that includes blockers"""
        blocker_updates = DataLoader.get_mock_updates_with_blockers()
        self.add_updates(blocker_updates)
    
    def clear_updates(self) -> None:
        """Clear all updates from the system"""
        self.updates = []
    
    def get_updates_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Update]:
        """Retrieve updates within a date range"""
        filtered_updates = []
        for update in self.updates:
            update_date = datetime.strptime(update.date, "%Y-%m-%d")
            if start_date <= update_date <= end_date:
                filtered_updates.append(update)
        return filtered_updates
    
    def get_updates_by_employee(self, employee_name: str) -> List[Update]:
        """Retrieve updates for a specific employee"""
        return [update for update in self.updates if update.employee == employee_name]
    
    def get_updates_by_role(self, role: str) -> List[Update]:
        """Retrieve updates for a specific role"""
        return [update for update in self.updates if update.role == role]
    
    def aggregate_updates(self, updates: Optional[List[Update]] = None) -> AggregatedData:
        """Aggregate and structure updates for report generation"""
        if updates is None:
            updates = self.updates
            
        # Group by role and extract key information
        role_summary = defaultdict(list)
        all_updates_text = []
        
        for update in updates:
            role_summary[update.role].append({
                "employee": update.employee,
                "update": update.update,
                "date": update.date
            })
            all_updates_text.append(f"{update.employee} ({update.role}): {update.update}")
        
        return AggregatedData(
            role_summary=dict(role_summary),
            all_updates=all_updates_text,
            total_updates=len(updates)
        )
    
    def generate_report(self, report_type: str = "weekly", date_range: Optional[Tuple[datetime, datetime]] = None, custom_prompt: Optional[str] = None) -> str:
        """Generate a report based on aggregated updates with improved prompt engineering and context management"""
        # Context management: filter updates by date range and limit to most recent 50
        if date_range:
            start_date, end_date = date_range
            updates = self.get_updates_by_date_range(start_date, end_date)
        else:
            updates = self.updates
        
        if not updates:
            return "No updates found for the specified criteria."
        
        # Limit to most recent 50 updates
        updates = sorted(updates, key=lambda u: u.date, reverse=True)[:50]
        
        # If too many updates, summarize by role
        if len(updates) > 30:
            # Group by role and summarize
            role_dict = defaultdict(list)
            for update in updates:
                role_dict[update.role].append(f"{update.employee}: {update.update}")
            context_lines = []
            for role, items in role_dict.items():
                context_lines.append(f"Role: {role}")
                context_lines.extend([f"- {item}" for item in items])
            context = "\n".join(context_lines)
        else:
            context = "\n".join([
                f"{update.employee} ({update.role}, {update.date}): {update.update}" for update in updates
            ])
        
        # Add context stats
        context_stats = f"[Context: {len(updates)} updates, {len(context)} characters]"
        
        # Few-shot example (short)
        example = (
            "# Example Report (Markdown)\n"
            "## Key Achievements\n"
            "- Frontend: Login flow completed, dashboard bugs fixed.\n"
            "- Backend: API optimized, payment integration started.\n"
            "\n## Current Focus\n"
            "- Payment UI, database migration, mobile app testing.\n"
            "\n## Blockers\n"
            "- PCI compliance for payments, vendor API delay.\n"
            "\n## Upcoming Priorities\n"
            "- Complete payment UI, finalize Q4 content, address mobile bugs.\n"
        )
        
        # Use custom prompt if provided, otherwise use default
        if custom_prompt:
            prompt = f"""
            {custom_prompt}
            
            {context_stats}
            
            Here are the team updates (delimited by triple backticks):
            ```
            {context}
            ```
            
            Please generate the report below:
            """
        else:
            # Default prompt engineering: explicit instructions, delimiters, example
            prompt = f"""
            You are an expert project manager AI assistant. Based on the following team updates, generate a **{report_type} report** for stakeholders.
            
            {context_stats}
            
            Please follow these instructions:
            - Use **Markdown** formatting with clear headings and bullet points.
            - Structure the report with the following sections:
              1. Key Achievements and Progress
              2. Current Focus Areas
              3. Blockers or Challenges
              4. Upcoming Priorities
            - Be concise, executive-friendly, and actionable.
            - Only use the information provided in the context.
            
            Here is an example of a good report:
            {example}
            
            ---
            
            Here are the team updates (delimited by triple backticks):
            ```
            {context}
            ```
            
            Please generate the report below:
            """
        report = self.llm.generate_response(prompt)
        return report
    
    def answer_stakeholder_question(self, question: str) -> str:
        """Answer specific questions from stakeholders"""
        if not self.updates:
            return "No team updates available to answer the question."
        
        # Create context from all updates
        context = "\n".join([
            f"{update.employee} ({update.role}, {update.date}): {update.update}"
            for update in self.updates
        ])
        
        prompt = f"""
        Based on the following team updates, please answer this stakeholder question:
        
        Question: {question}
        
        Team Updates:
        {context}
        
        Please provide a concise, data-driven answer.
        """
        
        answer = self.llm.generate_response(prompt)
        return answer
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics about the updates"""
        if not self.updates:
            return {
                "total_updates": 0,
                "team_members": 0,
                "roles": 0,
                "date_range": "No updates available"
            }
        
        df = pd.DataFrame([update.to_dict() for update in self.updates])
        
        return {
            "total_updates": len(self.updates),
            "team_members": df['employee'].nunique(),
            "roles": df['role'].nunique(),
            "date_range": f"{df['date'].min()} to {df['date'].max()}",
            "updates_by_role": df['role'].value_counts().to_dict(),
            "updates_by_employee": df['employee'].value_counts().to_dict()
        }
    
    def display_summary_stats(self) -> None:
        """Display summary statistics about the updates"""
        stats = self.get_summary_stats()
        
        print("=== TEAM UPDATES SUMMARY ===")
        print(f"Total Updates: {stats['total_updates']}")
        print(f"Team Members: {stats['team_members']}")
        print(f"Roles Represented: {stats['roles']}")
        print(f"Date Range: {stats['date_range']}")
        print()
        
        if stats['total_updates'] > 0:
            print("Updates by Role:")
            for role, count in stats['updates_by_role'].items():
                print(f"  {role}: {count} update(s)")
            print()
            
            print("Updates by Employee:")
            for employee, count in stats['updates_by_employee'].items():
                print(f"  {employee}: {count} update(s)")
            print()