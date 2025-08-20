# base_repository.py
"""Abstract base repository interface for update operations."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from core import Update


class BaseUpdateRepository(ABC):
    """Abstract base class for update repositories"""
    
    @abstractmethod
    def add(self, update: Update) -> None:
        """Add a single update"""
        pass
    
    @abstractmethod
    def add_many(self, updates: List[Update]) -> None:
        """Add multiple updates"""
        pass
    
    @abstractmethod
    def get_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Update]:
        """Get updates within date range"""
        pass
    
    @abstractmethod
    def get_by_employee(self, employee_name: str, limit: Optional[int] = None) -> List[Update]:
        """Get updates for specific employee"""
        pass
    
    @abstractmethod
    def get_by_role(self, role: str, limit: Optional[int] = None) -> List[Update]:
        """Get updates for specific role"""
        pass
    
    @abstractmethod
    def get_recent(self, limit: int = 50, offset: int = 0) -> List[Update]:
        """Get most recent updates"""
        pass
    
    @abstractmethod
    def count(self, employee_filter: Optional[str] = None, role_filter: Optional[str] = None) -> int:
        """Count total updates with optional filters"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all updates - USE WITH CAUTION"""
        pass
    
    @abstractmethod
    def get_all(self, limit: Optional[int] = None) -> List[Update]:
        """Get all updates - USE WITH CAUTION for large datasets"""
        pass