# Qwen3 Embedding Project Structure

## Core Files

### Models
- `qwen3-embedding-0.6b.gguf` - Optimized Q8_0 quantized Qwen3-Embedding model (610MB)
- `Modelfile` - Ollama model configuration (created during setup)

### API & Services  
- `qwen3-api.py` - OpenAI-compatible FastAPI wrapper for embeddings
- `qdrantsetup.py` - Optimized Qdrant vector store setup and management
- `requirements.txt` - Python dependencies

### Setup & Testing
- `setup.sh` - Automated setup script (recommended)
- `test_setup.py` - Comprehensive verification script
- `test_roocode_compatibility.py` - RooCode compatibility tests (if exists)
- `test_base64_conversion.py` - Base64 encoding tests (if exists)

### Configuration
- `README.md` - Complete setup and usage documentation
- `qdrant_storage/` - Qdrant database files
- `embedding_cache_0_6b/` - Cached embeddings for performance

## Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RooCode       │───▶│  Qwen3 API      │───▶│   Ollama        │
│   (Client)      │    │  :8000          │    │   :11434        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       ▼
         │                       │              qwen3-embedding
         │                       │              (610MB GGUF)
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   Qdrant        │◀───│  Vector Store   │
│   :6333         │    │  Management     │
└─────────────────┘    └─────────────────┘
```

## Ports & Endpoints

- **8000**: Qwen3 OpenAI-compatible API
- **6333**: Qdrant HTTP API  
- **6334**: Qdrant gRPC API
- **11434**: Ollama API

## RooCode Configuration

```yaml
Embeddings Provider: OpenAI-compatible
Base URL: http://localhost:8000
API Key: your-super-secret-qdrant-api-key
Model: qwen3
Embedding Dimension: 1024
Qdrant URL: http://localhost:6333
Qdrant API Key: your-super-secret-qdrant-api-key
Collection Name: qwen3_embedding
```

## Quick Commands

```bash
# Start everything
./setup.sh

# Test everything  
python test_setup.py

# Stop services
pkill -f qwen3-api.py
docker stop qdrant

# Restart individual services
python qwen3-api.py &
docker restart qdrant
ollama serve
```
