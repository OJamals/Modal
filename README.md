# RooCode Qwen3 Codebase Indexing Setup

Qwen3-embedding has topped embedding benchmarks, easily beating both open and close-source models. This project provides tools to **optimize any Qwen3-Embedding GGUF model** downloaded through Ollama, with an OpenAI-compatible API wrapper and optimized Qdrant vector store.

**🎯 Fully RooCode Compatible!** - Works seamlessly with [Cline](https://github.com/cline/cline) and its tributaries including Roo, KiloCode.

## Quick Start

**Automated Setup (Recommended)**
```bash
# One-command setup: downloads model, optimizes, and configures everything
./setup.sh
```

This automated script:
- Downloads Qwen3-Embedding-0.6B model (Q8_0-optimized) via Ollama
- Extracts and optimizes the GGUF model from Ollama storage  
- Creates optimized Ollama model for embedding-only usage
- Installs Python dependencies and starts all services
- Sets up Qdrant vector database with proper configuration
- Runs comprehensive tests to verify everything works

**Manual Setup (Advanced Users)**
```bash
# 1. Download and optimize Qwen3 model (0.6b:Q8 recommended, 4B:Q4 best)
ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
python optimize_gguf.py hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 qwen3-embedding

# 2. Install dependencies and start services
pip install -r requirements.txt
docker run -d --name qdrant -p 6333:6333 -e QDRANT__SERVICE__API_KEY="your-super-secret-qdrant-api-key" -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
python qwen3-api.py &
python qdrantsetup.py

# 3. Verify everything works
python test_setup.py
```

**Ready to use with RooCode!** The setup script displays the exact configuration values needed.

## What We Built

This setup provides a complete, optimized embedding pipeline with **Qwen developer recommendations**:

- **GGUF Model Optimizer**: `optimize_gguf.py` - Extracts and optimizes any Qwen3 model from Ollama
- **Instruction-Aware Embedding**: Task-specific instructions for 1-5% performance improvement  
- **MRL Support**: Matryoshka Representation Learning with 512, 768, and 1024 dimensions
- **OpenAI-Compatible API**: `qwen3-api.py` wrapper with RooCode base64 encoding support
- **Optimized Qdrant Vector Store**: `qdrantsetup.py` with performance tuning for 1024-dimensional vectors
- **Task-Specific Templates**: Code search, document retrieval, Q&A, clustering, and more
- **Complete RooCode Integration**: Ready-to-use with proper API keys and endpoints

### Qwen Developer Recommendations Implemented

✅ **Instruction-Aware Embedding**: Automatic task-specific instruction formatting  
✅ **MRL (Matryoshka Representation Learning)**: Custom dimension support (512/768/1024)  
✅ **Performance Optimization**: 1-5% improvement with tailored instructions  
✅ **Task-Specific Templates**: Optimized for different use cases  
✅ **Default Configurations**: Works out-of-the-box with optimal settings

## Services

- **Qwen3-0.6B API**: `http://localhost:8000` (Custom FastAPI wrapper, RooCode Compatible)
- **Qdrant Vector DB**: `http://localhost:6333` (Docker container with optimizations)
- **Ollama**: `http://localhost:11434` (Serving optimized GGUF model)

## Complete Setup Guide

### Automated Setup (Recommended)

The setup script handles everything automatically:

```bash
# Prerequisites: Docker and Ollama must be installed
# Docker: https://docs.docker.com/get-docker/
# Ollama: curl -fsSL https://ollama.ai/install.sh | sh

# Run the complete setup
./setup.sh

# The script automatically:
# 1. Downloads Qwen3-Embedding-0.6B (Q8_0) via Ollama
# 2. Extracts and optimizes the GGUF model from Ollama storage
# 3. Creates embedding-optimized Ollama model
# 4. Installs Python dependencies
# 5. Starts Qdrant vector database with proper configuration
# 6. Launches OpenAI-compatible API wrapper
# 7. Sets up optimized vector store collection
# 8. Runs comprehensive verification tests
# 9. Displays RooCode configuration values
```

### Manual Setup (Advanced Users)

For manual control over the process:

#### Step 1: Download and Optimize Model

```bash
# Download the Q8_0 quantized model (best quality/size balance)
ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0

# Extract and optimize from Ollama storage
python optimize_gguf.py hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 qwen3-embedding

# This creates:
# - qwen3-embedding-0.6b.gguf (optimized local copy)
# - Optimized Ollama model for embedding-only usage
```

#### Step 2: Start Services

```bash
# Install dependencies
pip install -r requirements.txt

# Start Qdrant vector database
docker run -d --name qdrant \
    -p 6333:6333 -p 6334:6334 \
    -e QDRANT__SERVICE__API_KEY="your-super-secret-qdrant-api-key" \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

# Start OpenAI-compatible API wrapper
python qwen3-api.py &
```

#### Step 3: Setup Vector Store

```bash
# Configure optimized Qdrant collection
python qdrantsetup.py

# Verify everything works
python test_setup.py
```

#### 2.4: Setup Qdrant Vector Store

```bash
# Start Qdrant with Docker
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -e QDRANT__SERVICE__API_KEY="your-super-secret-qdrant-api-key" -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Run optimized setup script
python qdrantsetup.py
# ✅ Creates optimized collection with HNSW indexing
# ✅ Tests embedding API connectivity  
# ✅ Adds sample documents for validation
```

## RooCode Integration

After running the setup script, you'll see the exact configuration values needed for RooCode integration:

```yaml
# RooCode Configuration (displayed by setup script)
Embeddings Provider: OpenAI-compatible
Base URL: http://localhost:8000
API Key: your-super-secret-qdrant-api-key
Model: qwen3-embedding
Embedding Dimension: 1024 # 4B upto 2560; 8B upto 4096

# Vector Database Configuration
Qdrant URL: http://localhost:6333
Qdrant API Key: your-super-secret-qdrant-api-key
Collection Name: qwen3_embedding
```

Simply copy these values into your RooCode settings after running `./setup.sh`.

## Usage Examples

### OpenAI-Compatible API with Qwen Features
```python
import requests

# Basic embedding (uses default "text_search" task)
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "Your text to embed",
    "model": "qwen3",
    "encoding_format": "float"
})

# Task-specific embedding (1-5% performance improvement)
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "model": "qwen3",
    "task": "code_search",  # Optimized for code
    "encoding_format": "float"
})

# Custom instruction embedding (maximum performance)
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "Advanced machine learning concepts",
    "model": "qwen3", 
    "instruction": "Represent this text for academic research and similarity:",
    "encoding_format": "float"
})

# MRL - Custom dimensions (Matryoshka Representation Learning)
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "Text for lower-dimensional embedding", 
    "model": "qwen3",
    "dimensions": 768,  # Instead of default 1024
    "encoding_format": "float"
})

embeddings = response.json()["data"][0]["embedding"]
print(f"Generated {len(embeddings)}-dimensional embedding")
```

### Task-Specific Instructions (Qwen Recommendation)
```python
# Available tasks with automatic instruction formatting:
tasks = [
    "text_search",      # General semantic search (default)
    "code_search",      # Code and programming tasks  
    "document_retrieval", # Document and text retrieval
    "question_answering", # Q&A systems
    "clustering",       # Text clustering and categorization
    "classification",   # Classification tasks
    "similarity",       # Semantic similarity comparison
    "general"          # General purpose embedding
]

# Each task automatically applies the optimal instruction format
# for 1-5% performance improvement as recommended by Qwen developers
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

## Verification & Testing

The setup script automatically runs comprehensive tests, but you can also run them manually:

```bash
# Complete system verification (run after setup)
python test_setup.py
# ✅ Tests all services and API endpoints
# ✅ Verifies embedding generation and vector storage
# ✅ Confirms RooCode compatibility
# ✅ Displays configuration values for easy copy-paste

# Test Qwen developer recommendations
python test_qwen_features.py
# ✅ Validates instruction-aware embedding
# ✅ Tests MRL support and dimension truncation
# ✅ Verifies task-specific instruction templates
# ✅ Confirms 1-5% performance optimization features

# Test individual components if needed
python qdrantsetup.py                 # Test Qdrant setup and indexing
curl http://localhost:8000/health      # API health check
curl http://localhost:6333/health      # Qdrant health check
curl http://localhost:11434/api/tags   # List Ollama models
```

## Performance Features

### Qwen Developer Recommendations (Implemented)

🚀 **Instruction-Aware Embedding**  
- Automatic task-specific instruction formatting
- 1-5% performance improvement over standard embedding
- 9 optimized instruction templates for different use cases

🎯 **MRL (Matryoshka Representation Learning)**  
- Support for 512, 768, and 1024 dimensions
- Smaller embeddings for faster search when full precision isn't needed
- Maintains quality with reduced dimensionality

⚡ **Optimized Configuration**  
- Embedding-only model setup (no text generation overhead)
- Memory-mapped file loading for faster startup
- Multi-threaded processing for better performance
- Optimal context window and rope frequency settings

📊 **Benchmarked Performance**  
- Q8_0 quantization: Best quality/size balance (610MB)
- 1024-dimensional embeddings with high semantic accuracy
- Fast inference optimized for codebase indexing workflows

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

- **Memory**: The Q8_0 model uses ~610MB
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