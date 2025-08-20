# exceptions.py
"""Custom exceptions for the PM reporting system."""


class ReportingError(Exception):
    """Base exception for reporting system errors"""
    pass


class QueryProcessingError(ReportingError):
    """Raised when query processing fails"""
    pass


class LLMProviderError(ReportingError):
    """Raised when LLM provider operations fail"""
    pass


class VectorServiceError(ReportingError):
    """Raised when vector service operations fail"""
    pass


class ConfigurationError(ReportingError):
    """Raised when configuration is invalid"""
    pass


class DatabaseError(ReportingError):
    """Raised when database operations fail"""
    pass
