"""
Docker utilities for managing Qdrant containers.
"""

import subprocess
import time
from typing import Optional

from qdrant_client import QdrantClient


def check_docker_connection(host: str = "localhost", port: int = 6333) -> bool:
    """
    Check if Docker Qdrant is accessible.
    
    Args:
        host: Docker host address
        port: Docker port number
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = QdrantClient(host=host, port=port)
        client.get_collections()
        return True
    except Exception:
        return False


def restart_docker_qdrant(host: str = "localhost", port: int = 6333) -> bool:
    """
    Restart Docker Qdrant when issues occur.
    
    Args:
        host: Docker host address  
        port: Docker port number
        
    Returns:
        True if restart was successful, False otherwise
    """
    print("🔄 Attempting to restart Docker Qdrant...")
    
    try:
        # Try to stop any existing qdrant containers
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", "ancestor=qdrant/qdrant"],
                capture_output=True, text=True, timeout=10
            )
            if result.stdout.strip():
                container_ids = result.stdout.strip().split('\n')
                for container_id in container_ids:
                    print(f"🛑 Stopping container {container_id}...")
                    subprocess.run(["docker", "stop", container_id], timeout=10)
        except Exception as e:
            print(f"⚠️  Could not stop existing containers: {e}")
        
        # Start fresh container
        print("🚀 Starting fresh Docker Qdrant...")
        subprocess.run([
            "docker", "run", "-d",  # detached
            "-p", f"{port}:{port}",
            "--name", f"qdrant-{int(time.time())}",  # unique name
            "qdrant/qdrant"
        ], timeout=30)
        
        # Wait for startup
        time.sleep(3)
        
        # Test connection
        if check_docker_connection(host, port):
            print("✅ Docker Qdrant restarted successfully!")
            return True
        else:
            print("❌ Docker restart failed - connection test failed")
            return False
            
    except Exception as e:
        print(f"❌ Docker restart failed: {e}")
        print(f"💡 Manual restart: docker run -p {port}:{port} qdrant/qdrant")
        return False


def get_docker_status(host: str = "localhost", port: int = 6333) -> dict:
    """
    Get status information about Docker Qdrant.
    
    Args:
        host: Docker host address
        port: Docker port number
        
    Returns:
        Dictionary with status information
    """
    status = {
        "connected": False,
        "collections_count": 0,
        "error": None
    }
    
    try:
        client = QdrantClient(host=host, port=port)
        collections = client.get_collections()
        status["connected"] = True
        status["collections_count"] = len(collections.collections)
    except Exception as e:
        status["error"] = str(e)
    
    return status 