# models.py
"""SQLAlchemy database models for the PM reporting system."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Index, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from core import settings

Base = declarative_base()


class UpdateModel(Base):
    """SQLAlchemy model for team updates with optimized indexes"""
    
    __tablename__ = "updates"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Core fields
    employee = Column(String(100), nullable=False)
    role = Column(String(50), nullable=False)  
    date = Column(String(10), nullable=False)  # YYYY-MM-DD format
    update = Column(Text, nullable=False)
    
    # Timestamps for auditing
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Performance indexes - CRITICAL for scalability
    __table_args__ = (
        # Most common queries: date range filtering
        Index('idx_updates_date', 'date'),
        Index('idx_updates_date_desc', 'date', postgresql_using='btree'),
        
        # Employee and role filtering
        Index('idx_updates_employee', 'employee'),
        Index('idx_updates_role', 'role'),
        
        # Composite indexes for common query patterns
        Index('idx_updates_employee_date', 'employee', 'date'),
        Index('idx_updates_role_date', 'role', 'date'),
        Index('idx_updates_date_employee', 'date', 'employee'),
        
        # Full-text search preparation (for future semantic improvements)
        # Index('idx_updates_update_text', 'update', postgresql_using='gin'),  # PostgreSQL only
    )
    
    def __repr__(self):
        return f"<UpdateModel(id={self.id}, employee='{self.employee}', role='{self.role}', date='{self.date}')>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'employee': self.employee,
            'role': self.role,
            'date': self.date,
            'update': self.update,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.DATABASE_URL
        self.engine = None
        self.SessionLocal = None
        self._initialize()
    
    def _initialize(self):
        """Initialize database engine and session factory"""
        # SQLite configuration for development
        connect_args = {}
        if "sqlite" in self.database_url:
            connect_args = {
                "check_same_thread": False,  # Allow multiple threads
                "timeout": 20  # Connection timeout
            }
        
        self.engine = create_engine(
            self.database_url,
            connect_args=connect_args,
            echo=settings.DEBUG,  # Log SQL queries in debug mode
            pool_pre_ping=True,   # Verify connections before use
            pool_recycle=3600     # Recycle connections every hour
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Create all tables with indexes"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
    
    def drop_tables(self):
        """Drop all tables - USE WITH CAUTION"""
        Base.metadata.drop_all(bind=self.engine)
    
    def get_table_stats(self):
        """Get database statistics for monitoring"""
        with self.get_session() as session:
            try:
                count = session.query(UpdateModel).count()
                latest_update = session.query(UpdateModel).order_by(
                    UpdateModel.created_at.desc()
                ).first()
                
                return {
                    "total_updates": count,
                    "latest_update": latest_update.created_at.isoformat() if latest_update else None,
                    "database_url": self.database_url.split("@")[-1] if "@" in self.database_url else self.database_url
                }
            except Exception as e:
                return {"error": str(e)}


# Global database manager instance
db_manager = DatabaseManager()


def get_database_session():
    """Dependency function for getting database sessions"""
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()