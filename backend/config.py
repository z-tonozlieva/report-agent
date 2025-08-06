# config.py
from enum import Enum
from typing import List

class ReportType(Enum):
    """Enum for different report types"""
    WEEKLY = "weekly"
    PROJECT_STATUS = "project status"
    BLOCKERS = "blockers"
    ACHIEVEMENTS = "achievements"
    CUSTOM = "custom"

class Config:
    """Configuration settings for the PM reporting tool"""
    
    # Report settings
    DEFAULT_REPORT_TYPE = ReportType.WEEKLY
    MAX_CONTEXT_LENGTH = 4000  # Maximum context length for LLM
    
    # Common stakeholder questions for testing
    SAMPLE_QUESTIONS = [
        "What's the status of the mobile app?",
        "Are there any current blockers?",
        "What are the main achievements this week?",
        "Which team members need additional support?",
        "What are the risks for the Q4 launch?",
        "How is the payment system integration progressing?",
        "What testing is still needed?",
        "Are we on track for our deadlines?"
    ]
    
    # Employee roles for validation
    VALID_ROLES = [
        "Frontend Developer",
        "Backend Developer",
        "Full Stack Developer",
        "Product Manager",
        "DevOps Engineer",
        "QA Engineer",
        "UX Designer",
        "UI Designer",
        "Marketing Manager",
        "Senior Developer",
        "Tech Lead",
        "Engineering Manager"
    ]
    
    # Date format for updates
    DATE_FORMAT = "%Y-%m-%d"
    
    # Display settings
    DISPLAY_SETTINGS = {
        "max_update_length_display": 150,  # Max characters to display in summaries
        "truncate_long_responses": True,
        "show_employee_count": True,
        "show_role_distribution": True
    }