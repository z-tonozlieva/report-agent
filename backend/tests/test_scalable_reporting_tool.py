# test_scalable_reporting_tool.py
"""Tests for ScalableReportingTool - Basic functionality only."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.models import Update
from data.models import DatabaseManager
from services.scalable_reporting_tool import ScalableReportingTool


@pytest.fixture
def temp_db():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        temp_db_path = tmp_file.name
    
    # Override the database URL for testing
    original_url = os.environ.get('DATABASE_URL')
    os.environ['DATABASE_URL'] = f'sqlite:///{temp_db_path}'
    
    try:
        # Initialize the test database
        db_manager = DatabaseManager(f'sqlite:///{temp_db_path}')
        db_manager.create_tables()
        yield temp_db_path
    finally:
        # Cleanup
        if original_url:
            os.environ['DATABASE_URL'] = original_url
        else:
            os.environ.pop('DATABASE_URL', None)
        Path(temp_db_path).unlink(missing_ok=True)


@pytest.fixture
def mock_llm():
    """Mock LLM interface."""
    llm = Mock()
    llm.invoke.return_value = "Mock LLM response"
    return llm


@pytest.fixture
def reporting_tool(temp_db, mock_llm):
    """Create a ScalableReportingTool instance with test database."""
    from data.sqlalchemy_repository import SQLAlchemyUpdateRepository
    from data.models import DatabaseManager
    
    # Create test-specific database manager and repository
    db_manager = DatabaseManager(f'sqlite:///{temp_db}')
    session = db_manager.get_session()
    test_repository = SQLAlchemyUpdateRepository(session=session)
    
    # Create the reporting tool with the test repository
    return ScalableReportingTool(llm=mock_llm, repository=test_repository)


@pytest.fixture
def sample_updates():
    """Sample updates for testing."""
    return [
        Update("John Doe", "Engineer", "2024-01-15", "Implemented authentication system"),
        Update("Jane Smith", "Manager", "2024-01-16", "Conducted team retrospective"),
        Update("Bob Wilson", "Engineer", "2024-01-17", "Fixed critical bug in payment processing"),
        Update("Alice Johnson", "Designer", "2024-01-18", "Completed new dashboard mockups"),
        Update("Charlie Brown", "QA", "2024-01-19", "Automated testing suite for API endpoints")
    ]


class TestScalableReportingTool:
    """Test cases for ScalableReportingTool - Basic functionality."""

    def test_add_single_update(self, reporting_tool):
        """Test adding a single update."""
        update = Update("John", "Engineer", "2024-01-15", "Test update")
        reporting_tool.add_update(update)
        
        with reporting_tool.repository as repo:
            recent = repo.get_recent(limit=1)
            assert len(recent) == 1
            assert recent[0].employee == "John"

    def test_add_multiple_updates(self, reporting_tool, sample_updates):
        """Test adding multiple updates."""
        reporting_tool.add_updates(sample_updates)
        
        with reporting_tool.repository as repo:
            recent = repo.get_recent(limit=10)
            assert len(recent) == 5
            employees = {u.employee for u in recent}
            assert "John Doe" in employees
            assert "Jane Smith" in employees

    def test_get_updates_by_employee(self, reporting_tool, sample_updates):
        """Test filtering updates by employee."""
        reporting_tool.add_updates(sample_updates)
        
        john_updates = reporting_tool.get_updates_by_employee("John Doe")
        assert len(john_updates) == 1
        assert john_updates[0].employee == "John Doe"
        assert "authentication" in john_updates[0].update

    def test_get_updates_by_role(self, reporting_tool, sample_updates):
        """Test filtering updates by role."""
        reporting_tool.add_updates(sample_updates)
        
        engineer_updates = reporting_tool.get_updates_by_role("Engineer")
        assert len(engineer_updates) == 2
        assert all(u.role == "Engineer" for u in engineer_updates)
        
        engineer_names = {u.employee for u in engineer_updates}
        assert engineer_names == {"John Doe", "Bob Wilson"}

    def test_clear_updates(self, reporting_tool, sample_updates):
        """Test clearing all updates."""
        reporting_tool.add_updates(sample_updates)
        
        # Verify updates were added
        with reporting_tool.repository as repo:
            recent = repo.get_recent(limit=10)
            assert len(recent) == 5
        
        # Clear updates
        reporting_tool.clear_updates()
        
        # Verify updates were cleared
        with reporting_tool.repository as repo:
            recent = repo.get_recent(limit=10)
            assert len(recent) == 0