# Services package
"""Business services for the PM reporting system."""

from .data_loader import DataLoader
from .scalable_reporting_tool import ScalableReportingTool

# Use Pinecone vector service (lightweight)
try:
    from .pinecone_vector_service import PineconeVectorService as VectorService
    __all__ = ["DataLoader", "ScalableReportingTool", "VectorService"]
except ImportError:
    # Pinecone dependencies not available - skip vector service
    __all__ = ["DataLoader", "ScalableReportingTool"]
