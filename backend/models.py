# models.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

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
            "update": self.update
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Update':
        return cls(
            employee=data["employee"],
            role=data["role"],
            date=data["date"],
            update=data["update"]
        )

@dataclass
class AggregatedData:
    """Represents aggregated team updates"""
    role_summary: Dict[str, List[Dict[str, str]]]
    all_updates: List[str]
    total_updates: int
    
    def get_context_string(self) -> str:
        """Convert aggregated data to string format for LLM context"""
        context = f"Team Updates Summary:\nTotal Updates: {self.total_updates}\n\nUpdates by Role:\n"
        
        for role, role_updates in self.role_summary.items():
            context += f"\n{role}:\n"
            for update_info in role_updates:
                context += f"  - {update_info['employee']}: {update_info['update']}\n"
        
        return context