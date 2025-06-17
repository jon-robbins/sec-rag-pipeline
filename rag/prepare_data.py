"""
Data Preparation Pipeline for SEC Vector Store
==============================================

This script orchestrates the complete data pipeline from raw SEC filings 
to a loaded vector store ready for queries.

Pipeline Steps:
1. Download SEC 10-K filings from Hugging Face (JanosAudran/financial-reports-sec)
2. Process into chunked format OR full documents 
3. Generate embeddings (if not cached)
4. Load into vector store

Data Sources:
- Raw: Hugging Face dataset with sentence-level SEC filing data
- Processed: data/chunks.pkl (sentence chunks) + embeddings/chunks_embeddings.pkl
- Alternative: data/df_filings_full.parquet (full documents via DownloadCorpus)
"""

from pathlib import Path
from typing import List, Optional
import sys

def check_data_files() -> dict:
    """Check which data files are available."""
    files = {
        "chunks": Path("data/chunks.pkl"),
        "chunk_embeddings": Path("embeddings/chunks_embeddings.pkl"), 
        "full_parquet": Path("data/df_filings_full.parquet"),
        "duckdb": Path("data/sec_filings.duckdb"),
    }
    
    status = {}
    for name, path in files.items():
        status[name] = {
            "exists": path.exists(),
            "path": str(path),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 1) if path.exists() else 0
        }
    
    return status

def show_data_pipeline_status():
    """Show current status of the data pipeline."""
    print("🔍 SEC Vector Store Data Pipeline Status")
    print("=" * 50)
    
    files = check_data_files()
    
    print("\n📁 Data Files:")
    for name, info in files.items():
        status = "✅" if info["exists"] else "❌"
        size = f"({info['size_mb']} MB)" if info["exists"] else ""
        print(f"   {status} {name}: {info['path']} {size}")
    
    # Check vector store status
    try:
        from .load_data import load_chunks_to_vectorstore
        from . import create_vector_store
        
        print("\n🗄️ Vector Store Status:")
        vs = create_vector_store(use_docker=False)
        vs.init_collection()
        data_info = vs.get_data_info()
        
        if data_info["status"] == "loaded":
            print(f"   ✅ Loaded with {data_info['points_count']} chunks")
            print(f"   📊 Tickers: {', '.join(data_info['data_summary']['tickers'])}")
            print(f"   📅 Years: {', '.join(map(str, data_info['data_summary']['fiscal_years']))}")
        else:
            print(f"   ❌ {data_info.get('message', 'Not loaded')}")
            
    except Exception as e:
        print(f"   ❌ Error checking vector store: {e}")
    
    print("\n🚀 Next Steps:")
    
    if files["chunks"]["exists"] and files["chunk_embeddings"]["exists"]:
        print("   ✅ Ready! Run: python -m rag.load_data")
    elif files["full_parquet"]["exists"]:
        print("   ⚠️  Full documents available, but no chunked data")
        print("   💡 You can use DocumentStore for full-document scenarios")
    else:
        print("   📥 No processed data found. Options:")
        print("   1. Run chunking pipeline (if you have raw data)")
        print("   2. Download via: python -m rag.data_processing.download_corpus")

def load_vector_store():
    """Load the vector store with available data."""
    files = check_data_files()
    
    if files["chunks"]["exists"]:
        print("📊 Loading chunked data to vector store...")
        from .load_data import load_chunks_to_vectorstore
        vs = load_chunks_to_vectorstore()
        print("✅ Vector store loaded successfully!")
        return vs
    else:
        print("❌ No chunked data found. Run preparation pipeline first.")
        return None

def main():
    """Main function for command-line usage."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "status":
            show_data_pipeline_status()
        elif command == "load":
            load_vector_store()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python -m rag.prepare_data [status|load]")
    else:
        show_data_pipeline_status()

if __name__ == "__main__":
    main() 