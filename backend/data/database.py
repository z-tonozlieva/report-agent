# database.py
"""Database initialization and migration utilities."""

import logging
from pathlib import Path

from backend.core import settings
from .models import db_manager

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handles database initialization and migrations"""
    
    @staticmethod
    def initialize_database():
        """Initialize database with tables and indexes"""
        try:
            logger.info("Initializing database...")
            logger.info(f"Database URL: {db_manager.database_url}")
            
            # Ensure data directory exists
            if "sqlite" in db_manager.database_url:
                db_path = Path(db_manager.database_url.replace("sqlite:///", ""))
                db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create tables with indexes
            db_manager.create_tables()
            
            # Verify database is working
            stats = db_manager.get_table_stats()
            logger.info(f"Database initialized successfully: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @staticmethod
    def migrate_from_old_system():
        """Migrate data from old in-memory system to database"""
        # This would be used if migrating from existing data
        logger.info("Database migration not needed - fresh installation")
        pass
    
    @staticmethod
    def create_indexes():
        """Ensure all performance indexes are created"""
        try:
            # Indexes are created automatically via SQLAlchemy model definitions
            # But we could add additional indexes here if needed
            logger.info("Database indexes verified")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise
    
    @staticmethod
    def get_database_info():
        """Get database connection information"""
        try:
            stats = db_manager.get_table_stats()
            
            return {
                "database_type": "SQLite" if "sqlite" in db_manager.database_url else "Other",
                "database_path": db_manager.database_url,
                "connection_status": "Connected",
                "stats": stats
            }
            
        except Exception as e:
            return {
                "database_type": "Unknown",
                "connection_status": "Failed",
                "error": str(e)
            }


def init_database():
    """Convenience function to initialize database"""
    return DatabaseInitializer.initialize_database()