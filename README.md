# RooCode Qwen3 Codebase Indexing Setup

Qwen3-embedding has topped embedding benchmarks, easily beating both open and close-source models. This project provides tools to **optimize any Qwen3-Embedding GGUF model** downloaded through Ollama, with an OpenAI-compatible API wrapper and optimized Qdrant vector store.

**🎯 Fully RooCode Compatible!** - Works seamlessly with [Cline](https://github.com/cline/cline) and its tributaries including Roo, KiloCode.

## Quick Start

**Option 1: Automated Setup (Recommended)**
```bash
# Download and optimize a Qwen3 model, then setup everything
./setup.sh
```

**Option 2: Manual Setup**
```bash
# 1. Download and optimize a Qwen3 model
ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
python optimize_gguf.py

# 2. Install dependencies and start API
pip install -r requirements.txt
python qwen3-api.py &

# 3. Start Qdrant and setup vector store
docker run -d --name qdrant -p 6333:6333 -e QDRANT__SERVICE__API_KEY="your-super-secret-qdrant-api-key" -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
python qdrantsetup.py

# 4. Verify everything works
python test_setup.py
```

**Ready to use with RooCode!** Use the configuration values shown in the test output.

## What We Built

This setup provides a complete, optimized embedding pipeline:

- **GGUF Model Optimizer**: `optimize_gguf.py` - Extracts and optimizes any Qwen3 model from Ollama
- **Ollama Integration**: Serves optimized models with automatic memory management  
- **OpenAI-Compatible API**: `qwen3-api.py` wrapper with RooCode base64 encoding support
- **Optimized Qdrant Vector Store**: `qdrantsetup.py` with performance tuning for 1024-dimensional vectors
- **Complete RooCode Integration**: Ready-to-use with proper API keys and endpoints

## Services

- **Qwen3-0.6B API**: `http://localhost:8000` (Custom FastAPI wrapper, RooCode Compatible)
- **Qdrant Vector DB**: `http://localhost:6333` (Docker container with optimizations)
- **Ollama**: `http://localhost:11434` (Serving optimized GGUF model)

## Complete Setup Guide

### Step 1: Download Qwen3 Model via Ollama

Choose and download a Qwen3 embedding model:

```bash
# Recommended: Q8_0 quantized (best quality/size balance)
ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0

# Alternative: Q4_K_M quantized (smaller, faster)  
ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q4_K_M

# Verify download
ollama list | grep qwen3
```

### Step 2: Optimize the GGUF Model

Extract and optimize the model from Ollama's storage:

```bash
# Interactive mode - shows available models
python optimize_gguf.py

# Or specify directly
python optimize_gguf.py "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0" "qwen3-embedding"

# This creates:
# - qwen3-embedding.gguf (optimized copy)
# - Modelfile (embedding-specific optimizations)  
# - qwen3-embedding model in Ollama
```

### Step 3: Start OpenAI-Compatible API Wrapper

Launch the optimized FastAPI wrapper:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API wrapper
python qwen3-api.py
# ✅ API running on http://localhost:8000
# ✅ RooCode compatible with base64 encoding
# ✅ 1024-dimensional embeddings
```

### Step 4: Setup Optimized Qdrant Vector Store

Start Qdrant and create optimized collection:

```bash
# Start Qdrant with Docker
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -e QDRANT__SERVICE__API_KEY="your-super-secret-qdrant-api-key" -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Run optimized setup script
python qdrantsetup.py
# ✅ Creates optimized collection with HNSW indexing
# ✅ Tests embedding API connectivity  
# ✅ Adds sample documents for validation
```

### Step 5: RooCode Integration Settings

Configure RooCode with these exact values:

```yaml
# Configuration
Embeddings Provider: OpenAI-compatible
Base URL: http://localhost:8000
API Key: your-super-secret-qdrant-api-key
Model: qwen3
Embedding Dimension: 1024
Qdrant URL: http://localhost:6333
Qdrant API Key: your-super-secret-qdrant-api-key
Collection Name: qwen3_embedding
```

## Usage Examples

### OpenAI-Compatible API
```python
import requests

# Generate embeddings using the optimized API
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "Your text to embed",
    "model": "qwen3",
    "encoding_format": "float"  # or "base64" for RooCode
})

embeddings = response.json()["data"][0]["embedding"]
print(f"Vector dimensions: {len(embeddings)}")  # Should be 1024
```

### Vector Storage with Optimized Qdrant
```python
from qdrantsetup import OptimizedQdrantVectorStore

# Initialize with your settings
vs = OptimizedQdrantVectorStore(
    qdrant_url="http://localhost:6333",
    qdrant_api_key="your-super-secret-qdrant-api-key",
    embedding_api_url="http://localhost:8000",
    collection_name="qwen3_embedding"
)

# Add documents (uses optimized batching)
vs.add_document("Python is a programming language", {"category": "tech"})

# Search with filtering
results = vs.search("What is Python?", filters={"category": "tech"})
```

### Test & Validation

```bash
# Comprehensive setup verification
python test_setup.py
# ✅ Tests all services and configuration
# ✅ Verifies RooCode compatibility
# ✅ Shows ready-to-use configuration values

# Test individual components
python test_roocode_compatibility.py  # RooCode compatibility
python qdrantsetup.py                 # Qdrant setup and performance

# Manual health checks
curl http://localhost:8000/health      # API health
curl http://localhost:6333/health      # Qdrant health
curl http://localhost:11434/api/tags   # Ollama models
```

## API Endpoints

- `POST /v1/embeddings` - Create embeddings (OpenAI compatible)
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /` - API information

## Troubleshooting

### Common Issues

**Ollama model not found:**
```bash
# Check if model exists
ollama list
# If not, recreate it
ollama create qwen3-embedding -f Modelfile
```

**API connection errors:**
```bash
# Check if services are running
python test_setup.py
# Restart API if needed
pkill -f qwen3-api.py
python qwen3-api.py &
```

**Qdrant connection issues:**
```bash
# Check Qdrant container
docker ps | grep qdrant
# Restart if needed
docker restart qdrant
```

**Permission errors with Docker:**
```bash
# Add user to docker group (Linux/macOS)
sudo usermod -aG docker $USER
# Or run with sudo
sudo docker run -d --name qdrant ...
```

### Performance Tuning

- **Memory**: The Q8_0 model uses ~610MB RAM
- **CPU**: Embedding generation is CPU-intensive
- **Disk**: Qdrant stores vectors on disk, ensure sufficient space
- **Batch Size**: For large codebases, process in batches of 100-500 files

### Logs and Debugging

```bash
# Check API logs
python qwen3-api.py  # Run in foreground to see logs

# Check Qdrant logs
docker logs qdrant

# Check Ollama logs
ollama logs qwen3-embedding
```