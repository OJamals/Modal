# RooCode Qwen3 Codebase Indexing Setup

Qwen3-embedding has topped embedding benchmarks, easily beating both open and close-source models. This project provides tools to **optimize any Qwen3-Embedding GGUF model** downloaded through Ollama, with an OpenAI-compatible API wrapper and optimized Qdrant vector store.

**🎯 Fully RooCode Compatible!** - Works seamlessly with Cline tributaries Roo & KiloCode.

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

**Manual Setup**
```bash
# 1. Download and optimize Qwen3 model (0.6b:Q8 recommended, 4B:Q4 best)
ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
python optimize_gguf.py hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 qwen3-embedding

# 2. Install dependencies and start services
pip install -r requirements.txt
docker run -d --name qdrant -p 6333:6333 -e QDRANT__SERVICE__API_KEY="your-super-secret-qdrant-api-key" -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
python qdrantsetup.py
python qwen3-api.py
```

**Ready to use with RooCode!** The setup script displays the exact configuration values needed.

This setup provides a complete, optimized embedding pipeline with **Qwen developer recommendations**:

- **GGUF Model Optimizer**: `optimize_gguf.py` - Extracts and optimizes any Qwen3 model from Ollama
- **Instruction-Aware Embedding**: Task-specific instructions for 1-5% performance improvement  
- **MRL Support**: Matryoshka Representation Learning with 512, 768, and 1024 dimensions
- **OpenAI-Compatible API**: `qwen3-api.py` wrapper with RooCode base64 encoding support
- **Optimized Qdrant Vector Store**: `qdrantsetup.py` with performance tuning for 1024-dimensional vectors
- **Task-Specific Templates**: Code search, document retrieval, Q&A, clustering, and more
- **Complete RooCode Integration**: Ready-to-use with proper API keys and endpoints

## Services

- **Ollama**: `http://localhost:11434` (Serving optimized GGUF model)
- **Qwen3-0.6B API**: `http://localhost:8000` (OpenAI-compatible FastAPI wrapper, RooCode Compatible)
- **Qdrant Vector DB**: `http://localhost:6333` (Docker container with optimizations)

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

### Qwen3-Embedding Advanced Features

🚀 **Instruction-Aware Embedding**  
- Automatic task-specific instruction formatting
- 9 optimized instruction templates for different use cases

🎯 **MRL (Matryoshka Representation Learning)**  
- Support for 512, 768, and 1024 dimensions
- Smaller embeddings for faster search when full precision isn't needed
- Maintains quality with reduced dimensionality

⚡ **Optimized Configuration**  
- Memory-mapped file loading for faster startup (1-3ms load time)
- Multi-threaded processing for better performance
- Optimal context window and rope frequency settings

📊 **Benchmarked Performance**  
- Q8_0 quantization: Best quality/size balance (610MB)
- 1024-dimensional embeddings with high semantic accuracy
- Fast inference optimized for codebase indexing workflows

**Automatic instruction formatting**
tasks =
- "text_search",      # General semantic search (default)
- "code_search",      # Code and programming tasks  
- "document_retrieval", # Document and text retrieval
- "question_answering", # Q&A systems
- "clustering",       # Text clustering and categorization
- "classification",   # Classification tasks
- "similarity",       # Semantic similarity comparison
- "general"          # General purpose embedding

# Each task automatically applies the optimal instruction format

## API Endpoints

- `POST /v1/embeddings` - Create embeddings (OpenAI compatible)
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /` - API information

## Usage Examples

### OpenAI-Compatible Embedding API with Advanced Qwen Features
```python
import requests

# Basic embedding (uses default "text_search" task)
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "Your text to embed",
    "model": "qwen3-embedding",
    "encoding_format": "float"
})

# Task-specific embedding (1-5% performance improvement)
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "model": "qwen3-embedding",
    "task": "code_search",  # Optimized for code
    "encoding_format": "float"
})

# Custom instruction embedding (maximum performance)
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "Advanced machine learning concepts",
    "model": "qwen3-embedding", 
    "instruction": "Represent this text for academic research and similarity:",
    "encoding_format": "float"
})

# MRL - Custom dimensions (Matryoshka Representation Learning)
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "Text for lower-dimensional embedding", 
    "model": "qwen3-embedding",
    "dimensions": 768,  # Instead of default 1024
    "encoding_format": "float"
})

embeddings = response.json()["data"][0]["embedding"]
print(f"Generated {len(embeddings)}-dimensional embedding")
```


## Verification & Testing

```bash
# Test individual components if needed
curl http://localhost:8000/health      # API health check
curl http://localhost:6333/health      # Qdrant health check
curl http://localhost:11434/api/tags   # List Ollama models
