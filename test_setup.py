#!/usr/bin/env python3
"""
Test script to verify that the entire Qwen3 embedding setup is working correctly.
This script checks:
1. Ollama is running and has the qwen3-embedding model
2. The OpenAI-compatible API wrapper is running
3. Qdrant is accessible and has the collection
4. End-to-end embedding and vector storage works
"""

import requests
import json
import time
import sys
from typing import Optional

def check_service(url: str, service_name: str, timeout: int = 5) -> bool:
    """Check if a service is running and accessible."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {service_name} is running at {url}")
            return True
        else:
            print(f"❌ {service_name} returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ {service_name} is not accessible at {url}")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ {service_name} timed out at {url}")
        return False
    except Exception as e:
        print(f"❌ Error checking {service_name}: {e}")
        return False

def check_ollama_model() -> bool:
    """Check if Ollama has the qwen3-embedding model loaded."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            qwen_models = [m for m in models if "qwen3-embedding" in m.get("name", "").lower()]
            if qwen_models:
                print(f"✅ Ollama has qwen3-embedding model: {qwen_models[0]['name']}")
                return True
            else:
                print("❌ qwen3-embedding model not found in Ollama")
                print("Available models:", [m.get("name") for m in models])
                return False
        else:
            print(f"❌ Failed to get Ollama models list: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking Ollama models: {e}")
        return False

def test_embedding_api() -> bool:
    """Test the OpenAI-compatible embedding API."""
    try:
        payload = {
            "input": "This is a test embedding request",
            "model": "qwen3",
            "encoding_format": "float"
        }
        response = requests.post(
            "http://localhost:8000/v1/embeddings",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            embedding = data["data"][0]["embedding"]
            if len(embedding) == 1024:
                print(f"✅ Embedding API working - generated {len(embedding)}-dimensional vector")
                return True
            else:
                print(f"❌ Embedding dimension mismatch: got {len(embedding)}, expected 1024")
                return False
        else:
            print(f"❌ Embedding API failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Error testing embedding API: {e}")
        return False

def check_qdrant_collection() -> bool:
    """Check if Qdrant has the qwen3_embedding collection."""
    try:
        response = requests.get(
            "http://localhost:6333/collections/qwen3_embedding",
            headers={"api-key": "your-super-secret-qdrant-api-key"},
            timeout=10
        )
        
        if response.status_code == 200:
            collection_info = response.json()["result"]
            print(f"✅ Qdrant collection 'qwen3_embedding' exists with {collection_info['vectors_count']} vectors")
            return True
        else:
            print(f"❌ Qdrant collection check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking Qdrant collection: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🔍 Testing Qwen3 Embedding Setup...")
    print("=" * 50)
    
    all_passed = True
    
    # Check basic services
    services = [
        ("http://localhost:11434", "Ollama"),
        ("http://localhost:8000/health", "Qwen3 API"),
        ("http://localhost:6333/health", "Qdrant")
    ]
    
    for url, name in services:
        if not check_service(url, name):
            all_passed = False
    
    print()
    
    # Check Ollama model
    if not check_ollama_model():
        all_passed = False
    
    print()
    
    # Test embedding API
    if not test_embedding_api():
        all_passed = False
    
    print()
    
    # Check Qdrant collection
    if not check_qdrant_collection():
        all_passed = False
    
    print()
    print("=" * 50)
    
    if all_passed:
        print("🎉 All tests passed! Your Qwen3 embedding setup is ready for RooCode.")
        print()
        print("RooCode Configuration:")
        print("- Embeddings Provider: OpenAI-compatible")
        print("- Base URL: http://localhost:8000")
        print("- API Key: your-super-secret-qdrant-api-key")
        print("- Model: qwen3")
        print("- Embedding Dimension: 1024")
        print("- Qdrant URL: http://localhost:6333")
        print("- Qdrant API Key: your-super-secret-qdrant-api-key")
        print("- Collection Name: qwen3_embedding")
        return 0
    else:
        print("❌ Some tests failed. Please check the setup instructions in README.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
