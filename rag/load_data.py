"""
Data loading utilities for the RAG pipeline.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any

from . import create_vector_store


def load_chunks_to_vectorstore(chunks_path: str = "data/chunks.pkl", embeddings_path: str = "embeddings/chunks_embeddings.pkl"):
    """Load chunks from pickle file into the vector store."""
    
    # Load chunks
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    
    print(f"📊 Loaded {len(chunks)} chunks from {chunks_path}")
    
    # Load cached embeddings
    embeddings_path = Path(embeddings_path)
    if embeddings_path.exists():
        with open(embeddings_path, "rb") as f:
            embeddings_data = pickle.load(f)
        
        # Handle different embedding file formats
        if isinstance(embeddings_data, tuple):
            # Tuple format: (chunk_ids, embeddings, metadata)
            embeddings = embeddings_data[1]  # Use index 1 for actual embeddings
        elif isinstance(embeddings_data, list):
            # Direct list of embeddings
            embeddings = embeddings_data
        else:
            raise ValueError(f"Unexpected embeddings format: {type(embeddings_data)}")
            
        print(f"🎯 Loaded {len(embeddings)} cached embeddings from {embeddings_path}")
        
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
    else:
        print(f"⚠️ No cached embeddings found at {embeddings_path}")
        print("🔄 This will take a while to compute embeddings...")
        embeddings = None
    
    # Create vector store
    vs = create_vector_store(use_docker=False)
    vs.init_collection()
    
    # Prepare data for upsert
    texts = [chunk.text for chunk in chunks]
    metas = []
    ids = []
    
    # Use integer IDs and store original string IDs in metadata
    for i, chunk in enumerate(chunks):
        # Add original ID to metadata
        meta = chunk.metadata.copy()
        meta['original_id'] = chunk.id  # Store original string ID
        metas.append(meta)
        ids.append(i)  # Use integer ID for Qdrant
    
    print(f"🔄 Upserting {len(chunks)} chunks to vector store...")
    
    # Upsert with pre-computed embeddings if available
    if embeddings:
        vs.upsert(
            texts=texts,
            metas=metas,
            ids=ids,
            vectors=embeddings,
            batch_size=500  # Can use larger batch size with pre-computed vectors
        )
    else:
        vs.upsert(
            texts=texts,
            metas=metas,
            ids=ids,
            batch_size=20  # Smaller batch size for embedding computation
        )
    
    # Verify
    status = vs.get_status()
    print(f"✅ Vector store loaded successfully!")
    print(f"   Points count: {status.get('points_count', 0)}")
    
    return vs


if __name__ == "__main__":
    load_chunks_to_vectorstore() 