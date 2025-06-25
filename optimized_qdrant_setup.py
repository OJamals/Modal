#!/usr/bin/env python3
"""
Optimized Qdrant Vector Storage for Qwen3-0.6B Embeddings
Enhanced configuration for maximum performance with 1024-dimensional vectors
RooCode-compatible with API key support
"""

import requests
import time
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, 
    OptimizersConfigDiff, HnswConfigDiff, QuantizationConfig, ScalarQuantization
)
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedQdrantVectorStore:
    """
    Optimized Qdrant setup for Qwen3-0.6B embeddings
    - 1024 dimensions
    - Code indexing and text search
    - Performance optimizations
    - RooCode compatibility
    """
    
    def __init__(
        self, 
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str = "your-super-secret-qdrant-api-key",
        embedding_api_url: str = "http://localhost:8000",
        collection_name: str = "qwen3_embedding"
    ):
        # Initialize Qdrant client with API key
        # Note: Using HTTP for local development - consider HTTPS for production
        self.qdrant = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            # Suppress SSL warnings for local development
            verify=False if "localhost" in qdrant_url else True
        )
        self.embedding_api_url = embedding_api_url
        self.collection_name = collection_name
        self.vector_size = 1024  # Qwen3-0.6B embedding size
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from your RooCode-compatible Qwen3-0.6B API"""
        try:
            # Use RooCode-compatible request format
            response = requests.post(
                f"{self.embedding_api_url}/v1/embeddings",
                json={
                    "input": text,
                    "model": "qwen3",
                    "encoding_format": "float"  # Use float format for Qdrant
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract embedding from OpenAI-compatible response
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
            else:
                raise ValueError("Invalid response format from embedding API")
                
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise
    
    def create_optimized_collection(self):
        """Create collection with optimal settings for Qwen3-0.6B vectors"""
        try:
            # Check if collection exists and delete it to start fresh
            try:
                collection_info = self.qdrant.get_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' exists - deleting for fresh start")
                self.qdrant.delete_collection(self.collection_name)
                time.sleep(1)  # Wait for deletion to complete
            except:
                pass
            
            # Create collection with optimized configuration
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,  # Best for normalized embeddings
                    hnsw_config=HnswConfigDiff(
                        m=32,  # Higher for better recall (default: 16)
                        ef_construct=256,  # Higher for better index quality (default: 100)
                        full_scan_threshold=20000,  # Switch to exact search for small collections
                        max_indexing_threads=4,  # Parallel indexing
                        on_disk=False,  # Keep in memory for speed
                    )
                ),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,  # More segments for better performance
                    max_segment_size=200000,  # Larger segments for efficiency
                    memmap_threshold=100000,  # Use memory mapping for large datasets
                    indexing_threshold=50000,  # Start indexing after 50k vectors
                    flush_interval_sec=30,  # More frequent flushes
                    max_optimization_threads=2,  # Parallel optimization
                ),
                # Optional: Enable scalar quantization to reduce memory usage
                quantization_config=ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,  # Keep quantized vectors in RAM
                    )
                )
            )
            
            logger.info(f"✅ Created optimized collection: {self.collection_name}")
            logger.info("🔧 Optimizations enabled:")
            logger.info("   - HNSW index with m=32, ef_construct=256")
            logger.info("   - Cosine distance for normalized embeddings")
            logger.info("   - Memory-optimized configuration")
            logger.info("   - Scalar quantization (INT8) for memory efficiency")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def add_document(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """Add document with optimized batching"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        
        # Add useful metadata for filtering
        metadata.update({
            "text": text,
            "text_length": len(text),
            "indexed_at": time.time(),
            "word_count": len(text.split())
        })
        
        try:
            embedding = self.get_embedding(text)
            
            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload=metadata
            )
            
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"✅ Added document: {text[:50]}... (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    def add_documents_batch(
        self, 
        documents: List[Dict[str, Any]], 
        batch_size: int = 100
    ) -> List[str]:
        """Add multiple documents in optimized batches"""
        doc_ids = []
        
        # Process in batches for efficiency
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_points = []
            batch_ids = []
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            for doc in batch:
                doc_id = doc.get("id", str(uuid.uuid4()))
                text = doc["text"]
                metadata = doc.get("metadata", {})
                
                # Add standard metadata
                metadata.update({
                    "text": text,
                    "text_length": len(text),
                    "indexed_at": time.time(),
                    "word_count": len(text.split())
                })
                
                try:
                    embedding = self.get_embedding(text)
                    
                    point = PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload=metadata
                    )
                    
                    batch_points.append(point)
                    batch_ids.append(doc_id)
                    
                except Exception as e:
                    logger.error(f"Failed to process document {doc_id}: {e}")
                    continue
            
            # Upsert batch
            if batch_points:
                try:
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=batch_points
                    )
                    doc_ids.extend(batch_ids)
                    logger.info(f"✅ Added batch of {len(batch_points)} documents")
                    
                except Exception as e:
                    logger.error(f"Failed to upsert batch: {e}")
        
        return doc_ids
    
    def search(
        self, 
        query: str, 
        limit: int = 5,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Enhanced search with filtering and threshold"""
        try:
            query_embedding = self.get_embedding(query)
            
            # Build filter if provided
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Handle list values (e.g., category in ["tech", "docs"])
                        should_conditions = []
                        for v in value:
                            should_conditions.append(
                                FieldCondition(key=key, match=models.MatchValue(value=v))
                            )
                        conditions.append(Filter(should=should_conditions))
                    else:
                        conditions.append(
                            FieldCondition(key=key, match=models.MatchValue(value=value))
                        )
                
                if conditions:
                    # Use 'must' for AND logic, 'should' for OR logic within a field
                    search_filter = Filter(must=conditions)
            
            # Adjust score threshold when using filters (filtering can reduce similarity scores)
            effective_threshold = score_threshold * 0.1 if search_filter else score_threshold
            
            # Search with optimized parameters using query_points (new API)
            results = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=search_filter,
                limit=limit,
                score_threshold=effective_threshold,
                with_payload=True,
                with_vectors=False,  # Don't return vectors to save bandwidth
                search_params=models.SearchParams(
                    hnsw_ef=128,  # Higher ef for better recall during search
                    exact=False,  # Use approximate search for speed
                )
            )
            
            # Format results (query_points returns QueryResponse)
            formatted_results = []
            for result in results.points:
                payload = result.payload or {}
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"}
                })
            
            logger.info(f"🔍 Query: '{query}' - Found {len(formatted_results)} results" +
                       (f" (filtered by {filters})" if filters else ""))
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics and health info"""
        try:
            info = self.qdrant.get_collection(self.collection_name)
            
            # Get more detailed stats
            collection_info = {
                "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
                "vectors_count": info.vectors_count or 0,
                "points_count": info.points_count or 0,
                "indexed_vectors_count": getattr(info, 'indexed_vectors_count', 0),
                "collection_name": self.collection_name,
                "vector_size": self.vector_size,
                "config": {
                    "distance": "cosine",  # We know we set this to cosine
                    "size": self.vector_size
                }
            }
            
            # Try to get more accurate vector count from config
            if hasattr(info, 'config') and hasattr(info.config, 'params'):
                try:
                    if hasattr(info.config.params, 'vectors'):
                        vectors_config = info.config.params.vectors
                        if hasattr(vectors_config, 'size'):
                            collection_info["config"]["size"] = vectors_config.size
                        if hasattr(vectors_config, 'distance'):
                            collection_info["config"]["distance"] = str(vectors_config.distance).lower()
                except Exception:
                    pass  # Use defaults if we can't get the info
            
            return collection_info
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "error": str(e),
                "collection_name": self.collection_name,
                "vector_size": self.vector_size
            }

# Example usage and testing for RooCode compatibility
if __name__ == "__main__":
    print("🚀 Starting Optimized Qdrant Vector Store Setup for RooCode")
    print("=" * 60)
    
    # Initialize optimized vector store with RooCode compatibility
    vs = OptimizedQdrantVectorStore()
    
    # Test embedding API connectivity first
    print("\n🔌 Testing embedding API connectivity...")
    try:
        test_embedding = vs.get_embedding("test connection")
        print(f"✅ Embedding API connected - Vector size: {len(test_embedding)}")
        print(f"   Expected size: {vs.vector_size}")
        if len(test_embedding) != vs.vector_size:
            print(f"⚠️  Warning: Vector size mismatch!")
    except Exception as e:
        print(f"❌ Embedding API connection failed: {e}")
        print("   Make sure the Qwen3 API is running on http://localhost:8000")
        exit(1)
    
    # Create collection with optimizations
    print(f"\n🗄️  Creating optimized collection: {vs.collection_name}")
    vs.create_optimized_collection()
    
    # Test with code-related documents (RooCode use cases)
    test_documents = [
        {
            "text": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "metadata": {"category": "code", "language": "python", "type": "function", "complexity": "recursive"}
        },
        {
            "text": "FastAPI is a modern Python web framework for building APIs with automatic documentation and type hints",
            "metadata": {"category": "documentation", "topic": "web_frameworks", "language": "python", "framework": "fastapi"}
        },
        {
            "text": "Vector databases enable efficient similarity search over high-dimensional embeddings using algorithms like HNSW",
            "metadata": {"category": "documentation", "topic": "vector_search", "type": "concept", "algorithm": "hnsw"}
        },
        {
            "text": "class VectorStore: def __init__(self, client): self.client = client; self.index = None",
            "metadata": {"category": "code", "language": "python", "type": "class", "pattern": "initialization"}
        },
        {
            "text": "HNSW (Hierarchical Navigable Small World) graphs provide efficient approximate nearest neighbor search in high dimensions",
            "metadata": {"category": "documentation", "topic": "algorithms", "type": "explanation", "complexity": "advanced"}
        },
        {
            "text": "import numpy as np; from typing import List, Dict; def cosine_similarity(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))",
            "metadata": {"category": "code", "language": "python", "type": "function", "domain": "ml"}
        },
        {
            "text": "RooCode integration requires OpenAI-compatible embedding API with base64 encoding support for efficient vector transmission",
            "metadata": {"category": "documentation", "topic": "integration", "type": "requirement", "system": "roocode"}
        }
    ]
    
    # Add documents in batch
    print(f"\n📚 Adding {len(test_documents)} test documents...")
    doc_ids = vs.add_documents_batch(test_documents)
    print(f"✅ Successfully added {len(doc_ids)} documents")
    
    # Wait for indexing
    print("\n⏳ Waiting for Qdrant indexing...")
    time.sleep(3)
    
    # Test searches with RooCode-relevant queries
    test_queries = [
        "Python function for calculating fibonacci numbers",
        "How to build web APIs with FastAPI?", 
        "Vector similarity search algorithms HNSW",
        "Python class definition with initialization",
        "RooCode integration requirements",
        "machine learning cosine similarity calculation"
    ]
    
    print("\n🔍 Testing semantic search capabilities:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n� Query: '{query}'")
        results = vs.search(query, limit=3, score_threshold=0.3)  # Lowered threshold
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"   {i}. Score: {result['score']:.3f}")
                print(f"      Text: {result['text'][:70]}...")
                print(f"      Type: {result['metadata'].get('type', 'N/A')} | "
                      f"Category: {result['metadata'].get('category', 'N/A')}")
        else:
            print("   No results found above threshold")
    
    # Test filtered search (RooCode code indexing use case)
    print(f"\n🔍 Filtered search - Code only (RooCode use case):")
    results = vs.search(
        "Python programming functions and classes", 
        limit=5, 
        filters={"category": "code"}
    )
    
    for i, result in enumerate(results, 1):
        print(f"   {i}. Score: {result['score']:.3f}")
        print(f"      Text: {result['text'][:60]}...")
        print(f"      Language: {result['metadata'].get('language', 'N/A')} | "
              f"Type: {result['metadata'].get('type', 'N/A')}")
    
    # Show collection statistics
    stats = vs.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print("=" * 30)
    print(f"   Collection: {stats.get('collection_name', 'N/A')}")
    print(f"   Status: {stats.get('status', 'unknown')}")
    print(f"   Points: {stats.get('points_count', 0)}")
    print(f"   Vectors: {stats.get('vectors_count', 0)}")
    print(f"   Vector Size: {stats.get('vector_size', 0)}")
    
    # Test RooCode-specific requirements
    print(f"\n🧪 RooCode Compatibility Tests:")
    print("=" * 35)
    
    # Test embedding API health
    try:
        health_response = requests.get(f"{vs.embedding_api_url}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Embedding API health check passed")
        else:
            print(f"⚠️  Embedding API health check returned: {health_response.status_code}")
    except:
        print("❌ Embedding API health check failed")
    
    # Test API info endpoint
    try:
        info_response = requests.get(f"{vs.embedding_api_url}/v1/models", timeout=5)
        if info_response.status_code == 200:
            print("✅ API info endpoint accessible")
        else:
            print(f"⚠️  API info endpoint returned: {info_response.status_code}")
    except:
        print("❌ API info endpoint failed")
    
    print(f"\n🎉 Setup Complete! Vector store ready for RooCode integration")
    print(f"   - Collection: {vs.collection_name}")
    print(f"   - API URL: {vs.embedding_api_url}")
    print(f"   - Documents indexed: {stats.get('points_count', 0)}")
    print(f"   - Vector dimensions: {vs.vector_size}")
    print(f"\n💡 To use with RooCode:")
    print(f"   1. Set embedding API URL to: {vs.embedding_api_url}")
    print(f"   2. Set Qdrant collection to: {vs.collection_name}")
    print(f"   3. Ensure API key matches: your-super-secret-qdrant-api-key")
