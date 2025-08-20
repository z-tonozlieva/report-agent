# repository.py
"""Repository pattern for database operations."""

from datetime import datetime
from typing import List

from ..core import Update
from ..core.exceptions import DatabaseError


class UpdateRepository:
    """Repository for Update database operations"""

    def __init__(self, session=None):
        """Initialize repository with optional database session"""
        self.session = session
        # For now, we'll use in-memory storage for compatibility
        # In a real implementation, this would use SQLAlchemy
        self._updates: List[Update] = []

    def add(self, update: Update) -> None:
        """Add a single update"""
        try:
            self._updates.append(update)
            # In real implementation: self.session.add(update); self.session.commit()
        except Exception as e:
            raise DatabaseError(f"Failed to add update: {str(e)}")

    def add_many(self, updates: List[Update]) -> None:
        """Add multiple updates"""
        try:
            self._updates.extend(updates)
            # In real implementation: self.session.add_all(updates); self.session.commit()
        except Exception as e:
            raise DatabaseError(f"Failed to add updates: {str(e)}")

    def get_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Update]:
        """Get updates within a date range - OPTIMIZED"""
        try:
            # Pre-parse dates once for efficiency
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Efficient filtering without parsing dates multiple times
            filtered_updates = []
            for update in self._updates:
                # String comparison is faster than date parsing
                if start_str <= update.date <= end_str:
                    filtered_updates.append(update)

            return filtered_updates

            # In real implementation with database:
            # return self.session.query(Update).filter(
            #     Update.date >= start_date.strftime('%Y-%m-%d'),
            #     Update.date <= end_date.strftime('%Y-%m-%d')
            # ).all()

        except Exception as e:
            raise DatabaseError(f"Failed to get updates by date range: {str(e)}")

    def get_by_employee(self, employee_name: str) -> List[Update]:
        """Get updates for a specific employee"""
        try:
            return [update for update in self._updates if update.employee == employee_name]
            # In real implementation:
            # return self.session.query(Update).filter(Update.employee == employee_name).all()
        except Exception as e:
            raise DatabaseError(f"Failed to get updates for employee {employee_name}: {str(e)}")

    def get_by_role(self, role: str) -> List[Update]:
        """Get updates for a specific role"""
        try:
            return [update for update in self._updates if update.role == role]
            # In real implementation:
            # return self.session.query(Update).filter(Update.role == role).all()
        except Exception as e:
            raise DatabaseError(f"Failed to get updates for role {role}: {str(e)}")

    def get_recent(self, limit: int = 50) -> List[Update]:
        """Get most recent updates"""
        try:
            # Sort by date descending and take limit
            sorted_updates = sorted(self._updates, key=lambda x: x.date, reverse=True)
            return sorted_updates[:limit]
            # In real implementation:
            # return self.session.query(Update).order_by(Update.date.desc()).limit(limit).all()
        except Exception as e:
            raise DatabaseError(f"Failed to get recent updates: {str(e)}")

    def count(self) -> int:
        """Count total updates"""
        try:
            return len(self._updates)
            # In real implementation: return self.session.query(Update).count()
        except Exception as e:
            raise DatabaseError(f"Failed to count updates: {str(e)}")

    def clear(self) -> None:
        """Clear all updates"""
        try:
            self._updates.clear()
            # In real implementation: self.session.query(Update).delete(); self.session.commit()
        except Exception as e:
            raise DatabaseError(f"Failed to clear updates: {str(e)}")

    def get_all(self) -> List[Update]:
        """Get all updates - use with caution for large datasets"""
        try:
            return self._updates.copy()
            # In real implementation: return self.session.query(Update).all()
        except Exception as e:
            raise DatabaseError(f"Failed to get all updates: {str(e)}")
