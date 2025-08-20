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

    # Performance settings
    MAX_UPDATES_IN_MEMORY = 1000  # Limit in-memory updates
    QUERY_TIMEOUT = 50  # seconds

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
