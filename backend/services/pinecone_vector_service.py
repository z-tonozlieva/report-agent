# pinecone_vector_service.py
"""Pinecone-based vector service for semantic search and document storage."""

import os
import hashlib
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PineconeVectorService:
    """Vector service using Pinecone for storage and OpenAI for embeddings."""
    
    def __init__(self):
        """Initialize Pinecone connection and SentenceTransformer model."""
        self.pc = None
        self.index = None
        self.embedding_model = None
        self.index_name = "reportagent-updates"
        self.dimension = 384  # all-MiniLM-L6-v2 dimension (lightweight model)
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Pinecone and SentenceTransformer clients."""
        try:
            # Initialize Pinecone
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                logger.error("PINECONE_API_KEY not found in environment")
                return
                
            self.pc = Pinecone(api_key=pinecone_api_key)
            
            # Initialize SentenceTransformer for free embeddings
            logger.info("Loading SentenceTransformer model (free embeddings)...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, good quality
            logger.info("SentenceTransformer model loaded successfully")
            
            # Create or connect to index
            self._ensure_index_exists()
            
            logger.info("Pinecone vector service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone vector service: {e}")
            self.pc = None
            self.index = None
            self.embedding_model = None
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )
                
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            self.index = None
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using SentenceTransformer (free)."""
        try:
            if not self.embedding_model:
                return []
                
            # Generate embedding using free SentenceTransformer
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return []
    
    def _create_document_id(self, update_data: Dict[str, Any]) -> str:
        """Create a unique ID for the update."""
        # Use employee, date, and content hash for unique ID
        content = f"{update_data.get('employee', '')}_{update_data.get('date', '')}_{update_data.get('update', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_update(self, update_data: Dict[str, Any]):
        """Add a single update to the vector database."""
        if not self.index or not self.embedding_model:
            logger.warning("Vector service not available - skipping update")
            return
            
        try:
            update_text = f"Employee: {update_data.get('employee', '')} | Date: {update_data.get('date', '')} | Update: {update_data.get('update', '')}"
            
            embedding = self._get_embedding(update_text)
            if not embedding:
                logger.error("Failed to get embedding for update")
                return
                
            doc_id = self._create_document_id(update_data)
            
            # Prepare metadata
            metadata = {
                "employee": str(update_data.get('employee', '')),
                "date": str(update_data.get('date', '')),
                "role": str(update_data.get('role', '')),
                "department": str(update_data.get('department', '')),
                "update_text": update_text[:1000],  # Pinecone metadata limit
                "content_preview": str(update_data.get('update', ''))[:500]
            }
            
            # Upsert to Pinecone
            self.index.upsert([(doc_id, embedding, metadata)])
            
            logger.debug(f"Added update to vector DB: {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to add update to vector DB: {e}")
    
    def add_updates_batch(self, updates: List[Dict[str, Any]]) -> int:
        """Add multiple updates in batch."""
        if not self.index or not self.embedding_model:
            logger.warning("Vector service not available - skipping batch")
            return 0
            
        try:
            vectors_to_upsert = []
            
            for update_data in updates:
                update_text = f"Employee: {update_data.get('employee', '')} | Date: {update_data.get('date', '')} | Update: {update_data.get('update', '')}"
                
                embedding = self._get_embedding(update_text)
                if not embedding:
                    continue
                    
                doc_id = self._create_document_id(update_data)
                
                metadata = {
                    "employee": str(update_data.get('employee', '')),
                    "date": str(update_data.get('date', '')),
                    "role": str(update_data.get('role', '')),
                    "department": str(update_data.get('department', '')),
                    "update_text": update_text[:1000],
                    "content_preview": str(update_data.get('update', ''))[:500]
                }
                
                vectors_to_upsert.append((doc_id, embedding, metadata))
            
            # Batch upsert (Pinecone handles batching efficiently)
            if vectors_to_upsert:
                self.index.upsert(vectors_to_upsert)
                
            logger.info(f"Added {len(vectors_to_upsert)} updates to Pinecone")
            return len(vectors_to_upsert)
            
        except Exception as e:
            logger.error(f"Failed to batch add updates: {e}")
            return 0
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for semantically similar updates."""
        if not self.index or not self.embedding_model:
            logger.warning("Vector service not available - returning empty results")
            return []
            
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                return []
                
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=limit,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                metadata = match.metadata
                formatted_results.append({
                    "employee": metadata.get("employee", ""),
                    "date": metadata.get("date", ""),
                    "role": metadata.get("role", ""),
                    "department": metadata.get("department", ""),
                    "update": metadata.get("content_preview", ""),
                    "similarity_score": float(match.score)
                })
                
            logger.info(f"Found {len(formatted_results)} similar updates for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []
    
    def clear_collection(self):
        """Clear all vectors from the collection."""
        if not self.index:
            logger.warning("Vector service not available")
            return
            
        try:
            # Pinecone doesn't have a direct "clear all" - delete by fetching all IDs
            # For now, we'll delete the index and recreate it (simpler for small datasets)
            logger.info("Clearing Pinecone collection by deleting all vectors")
            self.index.delete(delete_all=True)
            logger.info("Pinecone collection cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        if not self.index:
            return {"status": "unavailable", "total_vectors": 0}
            
        try:
            stats = self.index.describe_index_stats()
            return {
                "status": "available",
                "total_vectors": stats.total_vector_count,
                "dimension": self.dimension,
                "index_name": self.index_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"status": "error", "total_vectors": 0}