# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Update:
    """Represents a single employee update"""

    employee: str
    role: str
    date: str
    update: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "employee": self.employee,
            "role": self.role,
            "date": self.date,
            "update": self.update,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Update":
        return cls(
            employee=data["employee"],
            role=data["role"],
            date=data["date"],
            update=data["update"],
        )


@dataclass
class AggregatedData:
    """Represents aggregated team updates"""

    role_updates: Dict[str, List[Update]]
    projects: List[str]
    achievements: List[str]  
    blockers: List[str]
    total_updates: int

    def get_context_string(self) -> str:
        """Convert aggregated data to string format for LLM context"""
        context = f"Team Updates Summary:\nTotal Updates: {self.total_updates}\n\nUpdates by Role:\n"

        for role, updates in self.role_updates.items():
            context += f"\n{role}:\n"
            for update in updates:
                context += f"  - {update.employee}: {update.update}\n"

        if self.projects:
            context += f"\nKey Projects: {', '.join(self.projects)}\n"
        
        if self.achievements:
            context += f"\nAchievements:\n"
            for achievement in self.achievements[:5]:  # Limit for brevity
                context += f"  - {achievement}\n"
        
        if self.blockers:
            context += f"\nBlockers:\n"
            for blocker in self.blockers[:5]:  # Limit for brevity
                context += f"  - {blocker}\n"

        return context
