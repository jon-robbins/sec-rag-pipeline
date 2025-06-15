"""
Collection management for the SEC Vector Store.
"""

import time
import warnings
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient, models

from .config import VectorStoreConfig
from .docker_utils import restart_docker_qdrant


class CollectionManager:
    """Handles Qdrant collection operations and data upserts."""
    
    def __init__(
        self, 
        client: QdrantClient, 
        config: VectorStoreConfig,
        embedding_manager=None
    ):
        self.client = client
        self.config = config
        self.embedding_manager = embedding_manager
        
    def init_collection(self) -> None:
        """
        (Re)create the collection with Docker-friendly error handling.
        """
        try:
            # Check if collection exists
            if self.client.collection_exists(self.config.collection_name):
                if self.config.use_docker:
                    print(f"🗑️  Attempting to delete existing collection '{self.config.collection_name}' in Docker...")
                
                try:
                    # Try to delete the collection
                    self.client.delete_collection(self.config.collection_name)
                    print(f"✅ Successfully deleted collection '{self.config.collection_name}'")
                except Exception as e:
                    print(f"⚠️  Failed to delete collection: {e}")
                    if self.config.use_docker:
                        print("🔄 Docker collection deletion failed - trying alternative approach...")
                        # Try to recreate with a different name
                        original_name = self.config.collection_name
                        self.config.collection_name = f"{original_name}_v{int(time.time())}"
                        print(f"🔄 Using new collection name: {self.config.collection_name}")
                    else:
                        raise  # Re-raise for in-memory mode

            # Create the collection
            print(f"🏗️  Creating collection '{self.config.collection_name}'...")
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.dim,
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"✅ Collection '{self.config.collection_name}' created successfully")
            
        except Exception as e:
            print(f"❌ Collection initialization failed: {e}")
            if self.config.use_docker and self.config.auto_fallback_to_memory:
                print("🔄 Docker collection failed - automatically switching to memory mode...")
                print("💡 Docker has persistent storage conflicts - memory mode avoids this issue")
                
                # Switch to memory mode
                print("🧠 Switching to memory mode...")
                self.config.use_docker = False
                self.client = QdrantClient(":memory:")
                
                # Retry collection initialization in memory mode
                try:
                    self.client.create_collection(
                        collection_name=self.config.collection_name,
                        vectors_config=models.VectorParams(
                            size=self.config.dim,
                            distance=models.Distance.COSINE,
                        ),
                    )
                    print(f"✅ Collection '{self.config.collection_name}' created successfully in memory mode")
                    return
                except Exception as memory_e:
                    print(f"❌ Memory mode also failed: {memory_e}")
                    raise
                    
            elif self.config.use_docker:
                print("💡 Docker collection failed. Try setting auto_fallback_to_memory=True or restart Docker:")
                print("   docker stop <container_id> && docker run -p 6333:6333 qdrant/qdrant")
            raise
    
    def upsert(
        self,
        *,
        metas: List[Dict[str, Any]],
        texts: Optional[List[str]] = None,
        vectors: Optional[List[List[float]]] = None,
        ids: Optional[List[Any]] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Upsert vectors and metadata into the collection.
        
        Args:
            metas: List of metadata dictionaries
            texts: Optional list of text strings (will be embedded if vectors not provided)
            vectors: Optional list of embedding vectors
            ids: Optional list of IDs (auto-generated if not provided)
            batch_size: Optional batch size override
        """
        # Get embeddings if not provided
        if vectors is None:
            if texts is None:
                raise ValueError("Provide either `texts` or `vectors`.")
            if self.embedding_manager is None:
                raise ValueError("EmbeddingManager required when vectors not provided")
            vectors = self.embedding_manager.embed_texts(texts)
        elif texts is None:
            texts = [""] * len(vectors)

        n = len(vectors)
        assert len(metas) == len(texts) == n, "vectors / metas / texts length mismatch"

        ids = ids or list(range(n))
        assert len(ids) == n, "`ids` length mismatch"

        # Check for dimension changes
        if len(vectors[0]) != self.config.dim:
            warnings.warn("Vector dim changed – recreating collection")
            self.config.dim = len(vectors[0])
            self.init_collection()

        # Prepare payloads
        payloads = [{**m, "text": t} for m, t in zip(metas, texts)]
        
        # Determine batch size
        if batch_size is None:
            if self.config.use_docker and n > 500:
                batch_size = self.config.docker_batch_size
                print(f"🐳 Docker mode: Using smaller batch size ({batch_size}) for {n} points...")
            else:
                batch_size = self.config.memory_batch_size

        print(f"🔄 Upserting {n} points to {'Docker' if self.config.use_docker else 'memory'} Qdrant...")
        
        try:
            # For large datasets in Docker, use batched approach
            if self.config.use_docker and n > batch_size:
                print(f"📦 Processing in batches of {batch_size}...")
                
                for i in range(0, n, batch_size):
                    end_idx = min(i + batch_size, n)
                    batch_vectors = vectors[i:end_idx]
                    batch_payloads = payloads[i:end_idx]
                    batch_ids = ids[i:end_idx]
                    
                    points = [
                        models.PointStruct(id=batch_ids[j], vector=batch_vectors[j], payload=batch_payloads[j])
                        for j in range(len(batch_vectors))
                    ]
                    
                    print(f"  🔄 Batch {i//batch_size + 1}: Uploading {len(points)} points...")
                    
                    self.client.upsert(
                        collection_name=self.config.collection_name,
                        points=points,
                    )
                    
                    # Brief pause between batches to avoid overwhelming Docker
                    if i + batch_size < n:
                        time.sleep(0.1)
                        
                print("✅ All batches uploaded successfully!")
            else:
                # Single batch upload for smaller datasets or memory mode
                points = [
                    models.PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
                    for i in range(n)
                ]
                
                self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=points,
                )
            
            # Verify the upsert worked by checking count
            info = self.client.get_collection(self.config.collection_name)
            print(f"✅ Upsert successful, collection now has {info.points_count} points")
            
        except Exception as e:
            print(f"❌ Upsert failed: {e}")
            if self.config.use_docker and self.config.auto_fallback_to_memory:
                print("🔄 Docker upsert failed - automatically switching to memory mode...")
                print("💡 Tip: For large datasets, memory mode is often faster than Docker")
                
                # Automatic fallback to memory mode
                print("🧠 Switching to memory mode...")
                self.config.use_docker = False
                self.client = QdrantClient(":memory:")
                self.init_collection()
                
                # Retry the upsert in memory mode
                return self.upsert(
                    metas=metas, texts=texts, vectors=vectors, ids=ids, batch_size=batch_size
                )
            elif self.config.use_docker:
                print("💡 Docker upsert failed. Try restarting Docker Qdrant or set auto_fallback_to_memory=True")
            raise 