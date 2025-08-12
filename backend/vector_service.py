# vector_service.py
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from .models import Update

logger = logging.getLogger(__name__)


class VectorService:
    """Service for managing vector embeddings and semantic search"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the vector service with ChromaDB and embedding model"""
        self.persist_directory = persist_directory
        self.embedding_model = None
        self.client = None
        self.collection = None
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and embedding model"""
        try:
            # Initialize ChromaDB with persistence
            logger.info(
                f"Initializing ChromaDB with persist directory: {self.persist_directory}"
            )
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection for updates
            self.collection = self.client.get_or_create_collection(
                name="updates",
                metadata={"description": "Team status updates with embeddings"},
            )

            # Initialize embedding model (runs locally, no API costs)
            logger.info("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Vector service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector service: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text"""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")

        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def add_update(self, update: Update) -> bool:
        """Add an update to the vector database"""
        try:
            # Create a comprehensive text for embedding
            full_text = (
                f"{update.employee} ({update.role}) on {update.date}: {update.update}"
            )

            # Generate embedding
            embedding = self.generate_embedding(update.update)

            # Create unique ID
            # Create unique ID using content hash and timestamp
            content_hash = hashlib.md5(update.update.encode()).hexdigest()[:8]
            timestamp_hash = hashlib.md5(
                f"{update.employee}_{update.date}_{update.role}".encode()
            ).hexdigest()[:6]
            update_id = (
                f"{update.employee}_{update.date}_{content_hash}_{timestamp_hash}"
            )

            # Add to collection with metadata
            self.collection.add(
                embeddings=[embedding],
                documents=[update.update],
                metadatas=[
                    {
                        "employee": update.employee,
                        "role": update.role,
                        "date": update.date,
                        "full_text": full_text,
                    }
                ],
                ids=[update_id],
            )

            logger.info(f"Added update to vector DB: {update_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add update to vector DB: {str(e)}")
            return False

    def add_updates_batch(self, updates: List[Update]) -> int:
        """Add multiple updates to the vector database in batch"""
        success_count = 0

        try:
            embeddings = []
            documents = []
            metadatas = []
            ids = []

            for update in updates:
                try:
                    # Generate embedding
                    embedding = self.generate_embedding(update.update)
                    full_text = f"{update.employee} ({update.role}) on {update.date}: {update.update}"

                    # Create unique ID using content hash and timestamp
                    content_hash = hashlib.md5(update.update.encode()).hexdigest()[:8]
                    timestamp_hash = hashlib.md5(
                        f"{update.employee}_{update.date}_{update.role}".encode()
                    ).hexdigest()[:6]
                    update_id = f"{update.employee}_{update.date}_{content_hash}_{timestamp_hash}"

                    # Check if this ID is already in our batch to prevent duplicates
                    if update_id not in ids:
                        embeddings.append(embedding)
                        documents.append(update.update)
                        metadatas.append(
                            {
                                "employee": update.employee,
                                "role": update.role,
                                "date": update.date,
                                "full_text": full_text,
                            }
                        )
                        ids.append(update_id)
                    else:
                        logger.warning(
                            f"Skipping duplicate update ID in batch: {update_id}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to process update for {update.employee}: {str(e)}"
                    )
                    continue

            if embeddings:
                self.collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                )
                success_count = len(embeddings)
                logger.info(f"Added {success_count} updates to vector DB in batch")

        except Exception as e:
            logger.error(f"Batch add failed: {str(e)}")

        return success_count

    def semantic_search(
        self, query: str, limit: int = 10, filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search on updates"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)

            # Build where clause for filtering
            where_clause = {}
            if filters:
                if "employee" in filters:
                    where_clause["employee"] = filters["employee"]
                if "role" in filters:
                    where_clause["role"] = filters["role"]
                if "date_from" in filters and "date_to" in filters:
                    # ChromaDB doesn't support date range queries directly
                    # We'll filter after retrieval
                    pass

            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]

                    # Apply date filtering if specified
                    if filters and "date_from" in filters and "date_to" in filters:
                        update_date = datetime.strptime(metadata["date"], "%Y-%m-%d")
                        date_from = datetime.strptime(filters["date_from"], "%Y-%m-%d")
                        date_to = datetime.strptime(filters["date_to"], "%Y-%m-%d")

                        if not (date_from <= update_date <= date_to):
                            continue

                    formatted_results.append(
                        {
                            "document": doc,
                            "metadata": metadata,
                            "similarity_score": 1
                            - distance,  # Convert distance to similarity
                            "employee": metadata["employee"],
                            "role": metadata["role"],
                            "date": metadata["date"],
                        }
                    )

            logger.info(f"Semantic search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []

    def find_similar_updates(
        self, update_text: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find updates similar to the given update text"""
        return self.semantic_search(update_text, limit=limit)

    def get_themes(self, limit: int = 50) -> List[str]:
        """Extract common themes from recent updates (simplified approach)"""
        try:
            # Get recent updates
            results = self.collection.query(
                query_embeddings=[self.generate_embedding("general update")],
                n_results=limit,
                include=["documents"],
            )

            if not results or not results["documents"]:
                return []

            # Simple keyword extraction (can be enhanced with more sophisticated NLP)
            documents = results["documents"][0]
            themes = []

            common_keywords = [
                "payment",
                "integration",
                "testing",
                "mobile",
                "dashboard",
                "API",
                "database",
                "authentication",
                "UI",
                "UX",
                "frontend",
                "backend",
                "deployment",
                "performance",
                "bug",
                "feature",
            ]

            for keyword in common_keywords:
                count = sum(1 for doc in documents if keyword.lower() in doc.lower())
                if count >= 3:  # Threshold for theme
                    themes.append(f"{keyword.title()} ({count} mentions)")

            return themes

        except Exception as e:
            logger.error(f"Failed to extract themes: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            count = self.collection.count()
            return {
                "total_updates": count,
                "collection_name": self.collection.name,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """Clear all data from the collection"""
        try:
            # Delete the existing collection
            self.client.delete_collection(name="updates")

            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name="updates",
                metadata={"description": "Team status updates with embeddings"},
            )

            logger.info("Vector database cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False
