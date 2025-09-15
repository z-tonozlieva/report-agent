# settings.py
"""Centralized settings and configuration management."""

import os
from pathlib import Path


class Settings:
    """Application settings and configuration"""

    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    CONFIG_DIR = BASE_DIR / "config"
    WEB_DIR = BASE_DIR / "web"

    # Database
    DATABASE_URL = f"sqlite:///{DATA_DIR}/updates.db"

    # Vector database
    VECTOR_DB_PATH = str(DATA_DIR / "chroma_db")

    # LLM Configuration
    LLM_PROVIDER = "groq"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Web configuration
    STATIC_DIR = str(WEB_DIR / "static")
    TEMPLATES_DIR = str(WEB_DIR / "templates")

    # Application settings
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = ENVIRONMENT == "development"

    # Performance settings - Optimized for 512MB memory limit
    MAX_UPDATES_IN_MEMORY = 25   # Very aggressive limit for memory constraints
    QUERY_TIMEOUT = 50  # seconds
    
    # Memory optimization settings
    ENABLE_VECTOR_DB = os.getenv("ENABLE_VECTOR_DB", "false").lower() == "true"
    MAX_REPORT_UPDATES = 15  # Maximum updates for report generation - reduced further
    FORCE_GC_AFTER_REQUESTS = True  # Force garbage collection
    
    # Low memory mode for free tier deployment
    LOW_MEMORY_MODE = os.getenv("LOW_MEMORY_MODE", "true").lower() == "true"
    
    # Ultra-low memory settings for filtering
    MAX_FILTER_EMPLOYEES = 50   # Max employee names to show in filter
    MAX_FILTER_DEPARTMENTS = 20 # Max departments to show in filter

    # Entity configuration
    ENTITIES_CONFIG_PATH = str(CONFIG_DIR / "entities.yaml")

    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.CONFIG_DIR.mkdir(exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()
