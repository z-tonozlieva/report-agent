# test_sqlalchemy_repository_fixed.py
"""Tests for SQLAlchemy repository layer - Updated for actual interface."""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from core.models import Update
from data.models import UpdateModel, DatabaseManager
from data.sqlalchemy_repository import SQLAlchemyUpdateRepository


@pytest.fixture
def test_db_session():
    """Create a test database session that's isolated from production."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        temp_db_path = tmp_file.name
    
    try:
        # Create isolated database manager
        db_manager = DatabaseManager(f'sqlite:///{temp_db_path}')
        db_manager.create_tables()
        session = db_manager.get_session()
        
        yield session
        
    finally:
        session.close()
        Path(temp_db_path).unlink(missing_ok=True)


class TestSQLAlchemyUpdateRepository:
    """Test cases for SQLAlchemy repository operations."""

    def test_add_single_update(self, test_db_session):
        """Test adding a single update."""
        update = Update(
            employee="John Doe",
            role="Engineer", 
            date="2024-01-15",
            update="Completed feature X"
        )
        
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            repo.add(update)
        
        # Verify the update was added
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            recent = repo.get_recent(limit=1)
            assert len(recent) == 1
            assert recent[0].employee == "John Doe"
            assert recent[0].role == "Engineer"
            assert recent[0].update == "Completed feature X"

    def test_add_multiple_updates(self, test_db_session):
        """Test adding multiple updates."""
        updates = [
            Update("Alice", "Manager", "2024-01-15", "Team meeting"),
            Update("Bob", "Developer", "2024-01-16", "Bug fixes"),
            Update("Carol", "Designer", "2024-01-17", "UI mockups")
        ]
        
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            repo.add_many(updates)
        
        # Verify all updates were added
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            all_updates = repo.get_recent(limit=10)
            assert len(all_updates) == 3
            
            employees = {u.employee for u in all_updates}
            assert employees == {"Alice", "Bob", "Carol"}

    def test_get_recent_with_limit(self, test_db_session):
        """Test getting recent updates with limit."""
        # Add 5 updates
        updates = [
            Update("User1", "Role1", "2024-01-10", f"Update {i}")
            for i in range(5)
        ]
        
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            repo.add_many(updates)
            
            # Test limit functionality
            recent_3 = repo.get_recent(limit=3)
            assert len(recent_3) == 3
            
            recent_all = repo.get_recent(limit=10)
            assert len(recent_all) == 5

    def test_get_by_employee(self, test_db_session):
        """Test filtering updates by employee."""
        updates = [
            Update("John", "Engineer", "2024-01-15", "Update 1"),
            Update("Jane", "Manager", "2024-01-16", "Update 2"),
            Update("John", "Engineer", "2024-01-17", "Update 3")
        ]
        
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            repo.add_many(updates)
            
            john_updates = repo.get_by_employee("John")
            assert len(john_updates) == 2
            assert all(u.employee == "John" for u in john_updates)
            
            jane_updates = repo.get_by_employee("Jane")
            assert len(jane_updates) == 1
            assert jane_updates[0].employee == "Jane"

    def test_get_by_role(self, test_db_session):
        """Test filtering updates by role."""
        updates = [
            Update("John", "Engineer", "2024-01-15", "Code review"),
            Update("Jane", "Manager", "2024-01-16", "Team planning"),
            Update("Bob", "Engineer", "2024-01-17", "Bug fixing")
        ]
        
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            repo.add_many(updates)
            
            engineer_updates = repo.get_by_role("Engineer")
            assert len(engineer_updates) == 2
            assert all(u.role == "Engineer" for u in engineer_updates)
            
            manager_updates = repo.get_by_role("Manager")
            assert len(manager_updates) == 1
            assert manager_updates[0].role == "Manager"

    def test_get_by_date_range(self, test_db_session):
        """Test filtering updates by date range."""
        updates = [
            Update("User1", "Role1", "2024-01-10", "Before range"),
            Update("User2", "Role2", "2024-01-15", "Start of range"),
            Update("User3", "Role3", "2024-01-17", "In range"),
            Update("User4", "Role4", "2024-01-20", "End of range"),
            Update("User5", "Role5", "2024-01-25", "After range")
        ]
        
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            repo.add_many(updates)
            
            # Test date range filtering
            start_date = datetime(2024, 1, 15)
            end_date = datetime(2024, 1, 20)
            
            range_updates = repo.get_by_date_range(start_date, end_date)
            assert len(range_updates) == 3
            
            # Verify dates are within range
            for update in range_updates:
                update_date = datetime.strptime(update.date, "%Y-%m-%d")
                assert start_date <= update_date <= end_date

    def test_get_stats(self, test_db_session):
        """Test getting repository statistics."""
        updates = [
            Update("John", "Engineer", "2024-01-15", "Update 1"),
            Update("Jane", "Manager", "2024-01-16", "Update 2"),
            Update("Bob", "Engineer", "2024-01-17", "Update 3"),
            Update("Alice", "Designer", "2024-01-18", "Update 4")
        ]
        
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            repo.add_many(updates)
            
            stats = repo.get_stats()
            assert stats["total_updates"] == 4
            assert stats["role_distribution"]["Engineer"] == 2
            assert stats["role_distribution"]["Manager"] == 1
            assert stats["role_distribution"]["Designer"] == 1

    def test_search_updates(self, test_db_session):
        """Test searching updates by content."""
        updates = [
            Update("John", "Engineer", "2024-01-15", "Fixed critical bug in authentication"),
            Update("Jane", "Manager", "2024-01-16", "Team meeting about project timeline"),
            Update("Bob", "Engineer", "2024-01-17", "Implemented new authentication feature"),
            Update("Alice", "Designer", "2024-01-18", "Created mockups for dashboard")
        ]
        
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            repo.add_many(updates)
            
            # Search for authentication-related updates
            auth_updates = repo.search_updates("authentication")
            assert len(auth_updates) == 2
            assert all("authentication" in u.update.lower() for u in auth_updates)
            
            # Search for project-related updates
            project_updates = repo.search_updates("project")
            assert len(project_updates) == 1
            assert "project" in project_updates[0].update.lower()

    def test_empty_repository(self, test_db_session):
        """Test operations on empty repository."""
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            assert repo.get_recent() == []
            assert repo.get_by_employee("NonExistent") == []
            assert repo.get_by_role("NonExistent") == []
            assert repo.search_updates("anything") == []
            
            stats = repo.get_stats()
            assert stats["total_updates"] == 0
            assert stats["role_distribution"] == {}

    def test_count_method(self, test_db_session):
        """Test counting updates."""
        updates = [
            Update("John", "Engineer", "2024-01-15", "Update 1"),
            Update("Jane", "Manager", "2024-01-16", "Update 2"),
            Update("Bob", "Engineer", "2024-01-17", "Update 3")
        ]
        
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            repo.add_many(updates)
            
            # Test total count
            total_count = repo.count()
            assert total_count == 3
            
            # Test count with employee filter
            engineer_count = repo.count(role_filter="Engineer")
            assert engineer_count == 2
            
            # Test count with role filter
            john_count = repo.count(employee_filter="John")
            assert john_count == 1

    def test_performance_with_many_updates(self, test_db_session):
        """Test performance with larger dataset."""
        # Add a reasonable number of updates for testing
        updates = []
        for i in range(50):
            updates.append(Update(
                employee=f"Employee_{i % 10}",  # 10 unique employees
                role=f"Role_{i % 5}",          # 5 unique roles
                date=f"2024-01-{(i % 28) + 1:02d}",  # Various dates
                update=f"Update content {i}"
            ))
        
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            repo.add_many(updates)
            
            # These queries should be fast due to indexes
            employee_updates = repo.get_by_employee("Employee_0")
            assert len(employee_updates) == 5
            
            role_updates = repo.get_by_role("Role_0")
            assert len(role_updates) == 10
            
            # Date range query should also be efficient
            start_date = datetime(2024, 1, 10)
            end_date = datetime(2024, 1, 20)
            date_updates = repo.get_by_date_range(start_date, end_date, limit=50)
            assert len(date_updates) > 0

    def test_get_all_method(self, test_db_session):
        """Test get_all method."""
        updates = [
            Update("User1", "Role1", "2024-01-15", "Update 1"),
            Update("User2", "Role2", "2024-01-16", "Update 2"),
            Update("User3", "Role3", "2024-01-17", "Update 3")
        ]
        
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            repo.add_many(updates)
            
            # Test get_all without limit
            all_updates = repo.get_all()
            assert len(all_updates) == 3
            
            # Test get_all with limit
            limited_updates = repo.get_all(limit=2)
            assert len(limited_updates) == 2

    def test_get_by_employee_and_date_range(self, test_db_session):
        """Test composite filtering by employee and date range."""
        updates = [
            Update("John", "Engineer", "2024-01-10", "Early update"),
            Update("John", "Engineer", "2024-01-15", "Mid update"),
            Update("John", "Engineer", "2024-01-20", "Late update"),
            Update("Jane", "Manager", "2024-01-15", "Jane's update")
        ]
        
        with SQLAlchemyUpdateRepository(session=test_db_session) as repo:
            repo.add_many(updates)
            
            # Test composite filter
            start_date = datetime(2024, 1, 12)
            end_date = datetime(2024, 1, 18)
            
            john_updates = repo.get_by_employee_and_date_range(
                "John", start_date, end_date
            )
            assert len(john_updates) == 1
            assert john_updates[0].date == "2024-01-15"
            assert john_updates[0].employee == "John"