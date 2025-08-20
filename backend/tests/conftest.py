# conftest.py
"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.models import Update
from data.models import DatabaseManager


@pytest.fixture(scope="session")
def test_env():
    """Set up test environment variables."""
    # Store original values
    original_env = {}
    test_vars = {
        'ENVIRONMENT': 'test',
        'GROQ_API_KEY': 'test_key_12345',
    }
    
    for key, value in test_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield
    
    # Restore original values
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def temp_db():
    """Create a temporary test database for each test."""
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
    """Create a mock LLM interface."""
    llm = Mock()
    llm.invoke.return_value = "Mock LLM response for testing"
    return llm


@pytest.fixture
def mock_vector_service():
    """Create a mock vector service."""
    vector_service = Mock()
    vector_service.semantic_search.return_value = []
    vector_service.add_documents.return_value = None
    vector_service.update_documents.return_value = None
    return vector_service


@pytest.fixture
def sample_update():
    """Single sample update for testing."""
    return Update(
        employee="John Doe",
        role="Engineer",
        date="2024-01-15",
        update="Completed authentication system implementation"
    )


@pytest.fixture
def sample_updates():
    """Multiple sample updates for testing."""
    return [
        Update("John Doe", "Engineer", "2024-01-15", "Implemented authentication system with OAuth2"),
        Update("Jane Smith", "Manager", "2024-01-16", "Conducted team retrospective meeting"),
        Update("Bob Wilson", "Engineer", "2024-01-17", "Fixed critical bug in payment processing"),
        Update("Alice Johnson", "Designer", "2024-01-18", "Completed dashboard mockups for Q1"),
        Update("Charlie Brown", "QA", "2024-01-19", "Automated testing suite for API endpoints"),
        Update("Diana Ross", "Manager", "2024-01-20", "Coordinated cross-team collaboration"),
        Update("Eve Adams", "Engineer", "2024-01-21", "Optimized database query performance"),
        Update("Frank Castle", "Designer", "2024-01-22", "Designed mobile-responsive components"),
    ]


@pytest.fixture
def large_sample_updates():
    """Large dataset for performance testing."""
    updates = []
    roles = ["Engineer", "Manager", "Designer", "QA", "DevOps"]
    
    for i in range(100):
        updates.append(Update(
            employee=f"Employee_{i}",
            role=roles[i % len(roles)],
            date=f"2024-01-{(i % 28) + 1:02d}",
            update=f"Update content {i} - worked on feature development"
        ))
    
    return updates


@pytest.fixture
def sample_bulk_json():
    """Sample bulk upload JSON data."""
    return [
        {
            "employee": "Alice Cooper",
            "role": "Engineer", 
            "date": "2024-01-15",
            "update": "Implemented user authentication"
        },
        {
            "employee": "Bob Dylan",
            "role": "Designer",
            "date": "2024-01-16", 
            "update": "Created wireframes for new feature"
        },
        {
            "employee": "Carol King",
            "role": "Manager",
            "date": "2024-01-17",
            "update": "Planned sprint objectives"
        }
    ]


@pytest.fixture
def form_data_valid():
    """Valid form data for web tests."""
    return {
        "employee": "Test Employee",
        "role": "Engineer",
        "date": "2024-01-15",
        "update": "Completed test implementation"
    }


@pytest.fixture
def form_data_invalid():
    """Invalid form data for testing validation."""
    return {
        "employee": "",  # Missing required field
        "role": "Engineer",
        "date": "invalid-date",
        "update": "Test update"
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


# Custom collection hook for organizing tests
def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Add markers based on test file name
        if "test_web_api" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_query_handlers" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if any(keyword in item.nodeid.lower() for keyword in ["performance", "large", "slow"]):
            item.add_marker(pytest.mark.slow)