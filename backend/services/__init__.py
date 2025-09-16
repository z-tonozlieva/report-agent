# Services package
"""Business services for the PM reporting system."""

from .data_loader import DataLoader
from .scalable_reporting_tool import ScalableReportingTool

# Only import VectorService if chromadb is available
try:
    from .vector_service import VectorService
    __all__ = ["DataLoader", "ScalableReportingTool", "VectorService"]
except ImportError:
    # chromadb not available - skip vector service
    __all__ = ["DataLoader", "ScalableReportingTool"]
