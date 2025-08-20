# sqlalchemy_repository.py
"""SQLAlchemy-based repository implementation for production scalability."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_, desc, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core import Update
from core.exceptions import DatabaseError
from .base_repository import BaseUpdateRepository
from .models import UpdateModel, db_manager


class SQLAlchemyUpdateRepository(BaseUpdateRepository):
    """Production-ready SQLAlchemy repository with optimized queries and indexes"""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize with optional session, or create one"""
        self.session = session
        self._should_close_session = session is None
        
        if not self.session:
            self.session = db_manager.get_session()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # exc_type, exc_val, exc_tb are standard context manager parameters
        if self._should_close_session and self.session:
            self.session.close()
    
    def add(self, update: Update) -> None:
        """Add a single update to the database"""
        try:
            db_update = UpdateModel(
                employee=update.employee,
                role=update.role,
                date=update.date,
                update=update.update
            )
            self.session.add(db_update)
            self.session.commit()
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to add update: {str(e)}") from e
    
    def add_many(self, updates: List[Update]) -> None:
        """Add multiple updates efficiently using bulk operations"""
        try:
            db_updates = [
                UpdateModel(
                    employee=update.employee,
                    role=update.role,
                    date=update.date,
                    update=update.update
                )
                for update in updates
            ]
            
            # Use bulk insert for better performance
            self.session.bulk_save_objects(db_updates)
            self.session.commit()
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to add updates: {str(e)}") from e
    
    def get_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Update]:
        """Get updates within date range - OPTIMIZED with indexes"""
        try:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # This query uses idx_updates_date index for O(log n) performance
            query = self.session.query(UpdateModel).filter(
                and_(
                    UpdateModel.date >= start_str,
                    UpdateModel.date <= end_str
                )
            ).order_by(desc(UpdateModel.date))  # Most recent first
            
            if offset:
                query = query.offset(offset)
            
            if limit:
                query = query.limit(limit)
            
            db_updates = query.all()
            return [self._to_update_model(db_update) for db_update in db_updates]
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get updates by date range: {str(e)}") from e
    
    def get_by_employee(
        self, 
        employee_name: str, 
        limit: Optional[int] = None
    ) -> List[Update]:
        """Get updates for specific employee - OPTIMIZED with employee index"""
        try:
            # Uses idx_updates_employee index
            query = self.session.query(UpdateModel).filter(
                UpdateModel.employee == employee_name
            ).order_by(desc(UpdateModel.date))
            
            if limit:
                query = query.limit(limit)
            
            db_updates = query.all()
            return [self._to_update_model(db_update) for db_update in db_updates]
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get updates for employee {employee_name}: {str(e)}") from e
    
    def get_by_role(
        self, 
        role: str, 
        limit: Optional[int] = None
    ) -> List[Update]:
        """Get updates for specific role - OPTIMIZED with role index"""
        try:
            # Uses idx_updates_role index
            query = self.session.query(UpdateModel).filter(
                UpdateModel.role == role
            ).order_by(desc(UpdateModel.date))
            
            if limit:
                query = query.limit(limit)
            
            db_updates = query.all()
            return [self._to_update_model(db_update) for db_update in db_updates]
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get updates for role {role}: {str(e)}") from e
    
    def get_recent(self, limit: int = 50, offset: int = 0) -> List[Update]:
        """Get most recent updates - OPTIMIZED with date index"""
        try:
            # Uses idx_updates_date_desc index for fast DESC ordering
            db_updates = self.session.query(UpdateModel).order_by(
                desc(UpdateModel.date), desc(UpdateModel.id)
            ).offset(offset).limit(limit).all()
            
            return [self._to_update_model(db_update) for db_update in db_updates]
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get recent updates: {str(e)}") from e
    
    def get_by_employee_and_date_range(
        self,
        employee_name: str,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> List[Update]:
        """Get employee updates in date range - OPTIMIZED with composite index"""
        try:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Uses idx_updates_employee_date composite index
            query = self.session.query(UpdateModel).filter(
                and_(
                    UpdateModel.employee == employee_name,
                    UpdateModel.date >= start_str,
                    UpdateModel.date <= end_str
                )
            ).order_by(desc(UpdateModel.date))
            
            if limit:
                query = query.limit(limit)
            
            db_updates = query.all()
            return [self._to_update_model(db_update) for db_update in db_updates]
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get updates for employee in date range: {str(e)}") from e
    
    def search_updates(
        self,
        search_term: str,
        limit: int = 50,
        employee_filter: Optional[str] = None,
        role_filter: Optional[str] = None
    ) -> List[Update]:
        """Search updates by content - Basic text search"""
        try:
            query = self.session.query(UpdateModel)
            
            # Basic text search (case-insensitive)
            query = query.filter(UpdateModel.update.ilike(f"%{search_term}%"))
            
            # Apply filters
            if employee_filter:
                query = query.filter(UpdateModel.employee == employee_filter)
            
            if role_filter:
                query = query.filter(UpdateModel.role == role_filter)
            
            query = query.order_by(desc(UpdateModel.date)).limit(limit)
            
            db_updates = query.all()
            return [self._to_update_model(db_update) for db_update in db_updates]
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to search updates: {str(e)}") from e
    
    def count(self, employee_filter: Optional[str] = None, role_filter: Optional[str] = None) -> int:
        """Count total updates with optional filters"""
        try:
            query = self.session.query(func.count(UpdateModel.id))
            
            if employee_filter:
                query = query.filter(UpdateModel.employee == employee_filter)
            
            if role_filter:
                query = query.filter(UpdateModel.role == role_filter)
            
            return query.scalar()
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to count updates: {str(e)}") from e
    
    def clear(self) -> None:
        """Clear all updates - USE WITH CAUTION"""
        try:
            self.session.query(UpdateModel).delete()
            self.session.commit()
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to clear updates: {str(e)}") from e
    
    def get_all(self, limit: Optional[int] = None) -> List[Update]:
        """Get all updates - USE WITH CAUTION for large datasets"""
        try:
            query = self.session.query(UpdateModel).order_by(desc(UpdateModel.date))
            
            if limit:
                query = query.limit(limit)
            
            db_updates = query.all()
            return [self._to_update_model(db_update) for db_update in db_updates]
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get all updates: {str(e)}") from e
    
    def get_stats(self) -> dict:
        """Get repository statistics for monitoring"""
        try:
            total_count = self.session.query(func.count(UpdateModel.id)).scalar()
            
            # Count by role
            role_counts = self.session.query(
                UpdateModel.role, func.count(UpdateModel.id)
            ).group_by(UpdateModel.role).all()
            
            # Count by employee  
            employee_counts = self.session.query(
                UpdateModel.employee, func.count(UpdateModel.id)
            ).group_by(UpdateModel.employee).all()
            
            # Recent activity
            latest_update = self.session.query(UpdateModel).order_by(
                desc(UpdateModel.created_at)
            ).first()
            
            return {
                "total_updates": total_count,
                "role_distribution": dict(role_counts),
                "employee_counts": dict(employee_counts),
                "latest_update_date": latest_update.date if latest_update else None,
                "latest_update_created": latest_update.created_at.isoformat() if latest_update else None
            }
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get repository stats: {str(e)}") from e
    
    def _to_update_model(self, db_update: UpdateModel) -> Update:
        """Convert SQLAlchemy model to domain model"""
        return Update(
            employee=db_update.employee,
            role=db_update.role,
            date=db_update.date,
            update=db_update.update
        )