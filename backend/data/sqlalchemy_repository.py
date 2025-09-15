# sqlalchemy_repository.py
"""SQLAlchemy-based repository implementation for production scalability."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_, desc, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload

from backend.core import Update
from backend.core.exceptions import DatabaseError
from .base_repository import BaseUpdateRepository
from .models import UpdateModel, EmployeeModel, db_manager


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
    
    def _get_or_create_employee(self, update: Update) -> EmployeeModel:
        """Get existing employee or create new one"""
        employee = self.session.query(EmployeeModel).filter(
            EmployeeModel.name == update.employee
        ).first()
        
        if not employee:
            employee = EmployeeModel(
                name=update.employee,
                role=update.role,
                department=update.department,
                manager=update.manager
            )
            self.session.add(employee)
            self.session.flush()  # Get the ID without committing
        else:
            # Update employee info if provided
            if update.role and employee.role != update.role:
                employee.role = update.role
            if update.department and employee.department != update.department:
                employee.department = update.department
            if update.manager and employee.manager != update.manager:
                employee.manager = update.manager
                
        return employee
    
    def add(self, update: Update) -> None:
        """Add a single update to the database"""
        try:
            # Get or create employee
            employee = self._get_or_create_employee(update)
            
            # Create update linked to employee
            db_update = UpdateModel(
                employee_id=employee.id,
                date=update.date,
                update=update.update
            )
            self.session.add(db_update)
            self.session.commit()
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to add update: {str(e)}") from e
    
    def add_many(self, updates: List[Update]) -> None:
        """Add multiple updates efficiently"""
        try:
            for update in updates:
                # Get or create employee for each update
                employee = self._get_or_create_employee(update)
                
                # Create update linked to employee
                db_update = UpdateModel(
                    employee_id=employee.id,
                    date=update.date,
                    update=update.update
                )
                self.session.add(db_update)
            
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
            
            # Query with eager loading of employee data
            query = self.session.query(UpdateModel).options(
                joinedload(UpdateModel.employee_obj)
            ).filter(
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
        """Get updates for specific employee - OPTIMIZED with join"""
        try:
            query = self.session.query(UpdateModel).join(
                EmployeeModel
            ).options(
                joinedload(UpdateModel.employee_obj)
            ).filter(
                EmployeeModel.name == employee_name
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
        """Get updates for specific role - OPTIMIZED with join"""
        try:
            query = self.session.query(UpdateModel).join(
                EmployeeModel
            ).options(
                joinedload(UpdateModel.employee_obj)
            ).filter(
                EmployeeModel.role == role
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
            db_updates = self.session.query(UpdateModel).options(
                joinedload(UpdateModel.employee_obj)
            ).order_by(
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
        """Get employee updates in date range - OPTIMIZED with join"""
        try:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            query = self.session.query(UpdateModel).join(
                EmployeeModel
            ).options(
                joinedload(UpdateModel.employee_obj)
            ).filter(
                and_(
                    EmployeeModel.name == employee_name,
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
            query = self.session.query(UpdateModel).join(
                EmployeeModel
            ).options(
                joinedload(UpdateModel.employee_obj)
            )
            
            # Basic text search (case-insensitive)
            query = query.filter(UpdateModel.update.ilike(f"%{search_term}%"))
            
            # Apply filters
            if employee_filter:
                query = query.filter(EmployeeModel.name == employee_filter)
            
            if role_filter:
                query = query.filter(EmployeeModel.role == role_filter)
            
            query = query.order_by(desc(UpdateModel.date)).limit(limit)
            
            db_updates = query.all()
            return [self._to_update_model(db_update) for db_update in db_updates]
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to search updates: {str(e)}") from e
    
    def count(self, employee_filter: Optional[str] = None, role_filter: Optional[str] = None) -> int:
        """Count total updates with optional filters"""
        try:
            query = self.session.query(func.count(UpdateModel.id))
            
            if employee_filter or role_filter:
                query = query.join(EmployeeModel)
            
            if employee_filter:
                query = query.filter(EmployeeModel.name == employee_filter)
            
            if role_filter:
                query = query.filter(EmployeeModel.role == role_filter)
            
            return query.scalar()
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to count updates: {str(e)}") from e
    
    def clear(self) -> None:
        """Clear all updates and employees - USE WITH CAUTION"""
        try:
            # Delete updates first due to foreign key constraint
            self.session.query(UpdateModel).delete()
            # Delete employees 
            self.session.query(EmployeeModel).delete()
            self.session.commit()
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to clear updates: {str(e)}") from e
    
    def get_all(self, limit: Optional[int] = None) -> List[Update]:
        """Get all updates - USE WITH CAUTION for large datasets"""
        try:
            query = self.session.query(UpdateModel).options(
                joinedload(UpdateModel.employee_obj)
            ).order_by(desc(UpdateModel.date))
            
            if limit:
                query = query.limit(limit)
            
            db_updates = query.all()
            return [self._to_update_model(db_update) for db_update in db_updates]
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get all updates: {str(e)}") from e
    
    def get_stats(self) -> dict:
        """Get repository statistics for monitoring"""
        try:
            total_updates = self.session.query(func.count(UpdateModel.id)).scalar()
            total_employees = self.session.query(func.count(EmployeeModel.id)).scalar()
            
            # Count by role
            role_counts = self.session.query(
                EmployeeModel.role, func.count(UpdateModel.id)
            ).join(
                UpdateModel, EmployeeModel.id == UpdateModel.employee_id
            ).group_by(EmployeeModel.role).all()
            
            # Count by employee  
            employee_counts = self.session.query(
                EmployeeModel.name, func.count(UpdateModel.id)
            ).join(
                UpdateModel, EmployeeModel.id == UpdateModel.employee_id
            ).group_by(EmployeeModel.name).all()
            
            # Count by department
            department_counts = self.session.query(
                EmployeeModel.department, func.count(UpdateModel.id)
            ).join(
                UpdateModel, EmployeeModel.id == UpdateModel.employee_id
            ).filter(
                EmployeeModel.department.isnot(None)
            ).group_by(EmployeeModel.department).all()
            
            # Recent activity
            latest_update = self.session.query(UpdateModel).order_by(
                desc(UpdateModel.created_at)
            ).first()
            
            return {
                "total_updates": total_updates,
                "total_employees": total_employees,
                "role_distribution": dict(role_counts),
                "employee_counts": dict(employee_counts),
                "department_distribution": dict(department_counts),
                "latest_update_date": latest_update.date if latest_update else None,
                "latest_update_created": latest_update.created_at.isoformat() if latest_update else None
            }
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get repository stats: {str(e)}") from e
    
    def get_unique_roles(self) -> List[str]:
        """Get all unique roles from employees, ordered by usage frequency"""
        try:
            # Get roles ordered by frequency (most used first)
            role_counts = self.session.query(
                EmployeeModel.role, func.count(EmployeeModel.id)
            ).filter(
                EmployeeModel.role.isnot(None)
            ).group_by(EmployeeModel.role).order_by(
                func.count(EmployeeModel.id).desc()
            ).all()
            
            return [role for role, count in role_counts]
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get unique roles: {str(e)}") from e
    
    def get_all_employee_names(self) -> List[str]:
        """Get all employee names for autocomplete, ordered alphabetically"""
        try:
            names = self.session.query(EmployeeModel.name).order_by(
                EmployeeModel.name
            ).all()
            
            return [name[0] for name in names]
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get employee names: {str(e)}") from e
    
    def get_unique_departments(self) -> List[str]:
        """Get all unique departments, ordered by usage frequency"""
        try:
            # Get departments ordered by frequency (most used first)
            dept_counts = self.session.query(
                EmployeeModel.department, func.count(EmployeeModel.id)
            ).filter(
                EmployeeModel.department.isnot(None)
            ).group_by(EmployeeModel.department).order_by(
                func.count(EmployeeModel.id).desc()
            ).all()
            
            return [dept for dept, count in dept_counts]
            
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get unique departments: {str(e)}") from e
    
    def _to_update_model(self, db_update: UpdateModel) -> Update:
        """Convert SQLAlchemy model to domain model"""
        return Update(
            employee=db_update.employee_obj.name,
            role=db_update.employee_obj.role,
            department=db_update.employee_obj.department,
            manager=db_update.employee_obj.manager,
            date=db_update.date,
            update=db_update.update
        )