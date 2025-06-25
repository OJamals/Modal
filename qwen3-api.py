#!/usr/bin/env python3
"""
OpenAI-Compatible API Wrapper for Qwen3-Embedding-0.6B
Optimized for lightweight embedding tasks with 1024 dimensions
"""

import time
from typing import List, Dict, Any, Optional, Union
import numpy as np
import struct
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import requests
import hashlib
import json
import os

# Universal Configuration with Developer Recommendations
MODEL_CONFIG = {
    "model_name": "qwen3-embedding",  # Local optimized model
    "dimensions": 2560, # 4B = up to 2560, 8B = up to 4096
    "max_context_length": 32768,
    "temperature": 0.0,
    "supports_instructions": True,
    "supports_mrl": True,  # Matryoshka Representation Learning
    "available_dimensions": [512, 768, 1024],  # MRL supported dimensions
    "quantization": "Q8_0",
    "size_mb": 2500, # 4B:Q4 = 2500 MB, 8B:Q4 = 4700 MB
    "use_case": "Instruction-aware embedding with MRL support",
    "performance_improvement": "1-5% with task-specific instructions"
}

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request with Qwen3 enhancements"""
    input: Union[str, List[str]] = Field(..., description="Text to embed")
    model: str = Field(default=MODEL_CONFIG["model_name"], description="Model to use")
    encoding_format: str = Field(default="float", description="Encoding format (float or base64)")
    dimensions: Optional[int] = Field(
        default=MODEL_CONFIG["dimensions"], 
        description="Output dimensions (512, 768, or 1024 for MRL support)"
    )
    instruction: Optional[str] = Field(
        default=None, 
        description="Task-specific instruction for better performance (Qwen recommendation)"
    )
    task: Optional[str] = Field(
        default="text_search", 
        description="Task type for automatic instruction selection"
    )
    user: Optional[str] = Field(default=None, description="User ID")

class EmbeddingData(BaseModel):
    """Individual embedding data"""
    object: str = "embedding"
    embedding: Union[List[float], str]  # float array or base64 string
    index: int

class Usage(BaseModel):
    """Token usage information"""
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage

class Qwen3_0_6B_EmbeddingAPI:
    """Qwen3-Embedding-0.6B API wrapper with optimal configurations"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = MODEL_CONFIG["model_name"]
        self.dimensions = MODEL_CONFIG["dimensions"]
        self.cache_dir = "embedding_cache_0_6b"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
    
    def _save_to_cache(self, cache_key: str, embedding: List[float]):
        """Save embedding to cache"""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_path, 'w') as f:
                json.dump(embedding, f)
        except Exception:
            pass  # Cache errors shouldn't break the API
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Load embedding from cache"""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return None
    
    def _prepare_text_for_embedding(self, text: str, task: str = "text_search", custom_instruction: Optional[str] = None) -> str:
        """
        Prepare text with instruction-aware formatting (Qwen developer recommendation)
        Achieves 1-5% performance improvement with task-specific instructions
        """
        # Enhanced instruction mapping based on Qwen recommendations
        instruction_map = {
            "text_search": "Represent this text for semantic search and retrieval:",
            "code_search": "Represent this code for semantic search and similarity matching:",
            "code_indexing": "Represent this code for indexing and retrieval:",
            "document_retrieval": "Represent this document for retrieval and similarity search:",
            "question_answering": "Represent this text for question-answering tasks:",
            "clustering": "Represent this text for clustering and categorization:",
            "classification": "Represent this text for classification tasks:",
            "similarity": "Represent this text for semantic similarity comparison:",
            "general": "Represent this text for semantic understanding:"
        }
        
        if MODEL_CONFIG["supports_instructions"]:
            # Use custom instruction if provided, otherwise use task-specific instruction
            if custom_instruction:
                instruction = custom_instruction
            else:
                instruction = instruction_map.get(task, instruction_map["text_search"])
            
            # Qwen3-Embedding optimal format: Instruction + Text
            formatted_text = f"{instruction}\n{text}"
        else:
            # Fallback for non-instruction models
            formatted_text = text
            
        return formatted_text
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector (Ollama doesn't auto-normalize)"""
        np_embedding = np.array(embedding)
        norm = np.linalg.norm(np_embedding)
        if norm > 0:
            return (np_embedding / norm).tolist()
        return embedding
    
    def _encode_embedding_as_base64(self, embedding: List[float]) -> str:
        """Convert float array to base64 string (RooCode OpenAI-compatible format)"""
        # Convert to Float32Array (matching OpenAI's format)
        float32_array = np.array(embedding, dtype=np.float32)
        # Convert to bytes and then base64
        bytes_data = float32_array.tobytes()
        return base64.b64encode(bytes_data).decode('utf-8')
    
    def _apply_mrl_truncation(self, embedding: List[float], target_dimensions: int) -> List[float]:
        """
        Apply MRL (Matryoshka Representation Learning) truncation to custom dimensions
        Qwen3-Embedding supports 512, 768, and 1024 dimensions
        """
        if target_dimensions not in MODEL_CONFIG["available_dimensions"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported dimension {target_dimensions}. "
                       f"Supported dimensions: {MODEL_CONFIG['available_dimensions']}"
            )
        
        if target_dimensions >= len(embedding):
            return embedding  # No truncation needed
        
        # Truncate to target dimensions (MRL property)
        truncated = embedding[:target_dimensions]
        
        # Renormalize the truncated embedding
        np_embedding = np.array(truncated)
        norm = np.linalg.norm(np_embedding)
        if norm > 0:
            return (np_embedding / norm).tolist()
        return truncated

    def _generate_single_embedding(
        self, 
        text: str, 
        task: str = "text_search",
        custom_instruction: Optional[str] = None,
        target_dimensions: Optional[int] = None,
        use_cache: bool = True
    ) -> List[float]:
        """Generate embedding for a single text with Qwen developer recommendations"""
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(f"{text}|{task}|{custom_instruction}|{target_dimensions}")
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                return cached_embedding
        
        # Prepare text with instruction-aware formatting (1-5% improvement)
        formatted_text = self._prepare_text_for_embedding(text, task, custom_instruction)
        
        # Generate embedding via Ollama
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": formatted_text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                
                # Validate dimensions
                if len(embedding) != self.dimensions:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Expected {self.dimensions} dimensions, got {len(embedding)}"
                    )
                
                # Normalize embedding
                normalized_embedding = self._normalize_embedding(embedding)
                
                # Apply MRL truncation if requested (Qwen developer feature)
                if target_dimensions and MODEL_CONFIG["supports_mrl"]:
                    normalized_embedding = self._apply_mrl_truncation(normalized_embedding, target_dimensions)
                
                # Cache result
                if use_cache:
                    self._save_to_cache(cache_key, normalized_embedding)
                
                return normalized_embedding
            else:
                # Parse Ollama error response
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", response.text)
                except:
                    error_msg = response.text
                
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Ollama API error: {error_msg}"
                )
                
        except requests.RequestException as e:
            raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}")
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        encoding_format: str = "float",
        task: str = "text_search",
        custom_instruction: Optional[str] = None,
        target_dimensions: Optional[int] = None
    ) -> EmbeddingResponse:
        """Generate embeddings with Qwen developer recommendations (instruction-aware + MRL)"""
        start_time = time.time()
        embeddings = []
        total_tokens = 0
        
        for i, text in enumerate(texts):
            try:
                # Get embedding with instruction-aware formatting and MRL support
                float_embedding = self._generate_single_embedding(
                    text=text,
                    task=task,
                    custom_instruction=custom_instruction,
                    target_dimensions=target_dimensions
                )
                
                # Convert to requested format
                if encoding_format == "base64":
                    # RooCode expects base64-encoded Float32Array
                    embedding_data = self._encode_embedding_as_base64(float_embedding)
                else:
                    # Default float format
                    embedding_data = float_embedding
                
                embeddings.append(EmbeddingData(
                    embedding=embedding_data,
                    index=i
                ))
                # Rough token estimation (1 token ≈ 4 characters)
                total_tokens += len(text) // 4
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate embedding for text {i}: {str(e)}"
                )
        
        return EmbeddingResponse(
            data=embeddings,
            model=self.model_name,
            usage=Usage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )

# FastAPI app setup
app = FastAPI(
    title="qwen3",
    description="OpenAI-compatible API for Qwen3-Embedding-0.6B model",
    version="1.0.0"
)

# Initialize API
embedding_api = Qwen3_0_6B_EmbeddingAPI()

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Qwen3-Embedding-0.6B API - RooCode Compatible",
        "model": MODEL_CONFIG["model_name"],
        "dimensions": MODEL_CONFIG["dimensions"],
        "max_context": MODEL_CONFIG["max_context_length"],
        "use_case": MODEL_CONFIG["use_case"],
        "features": [
            "OpenAI-compatible /v1/embeddings endpoint",
            "Base64 encoding support (encoding_format: base64)",
            "Float array support (encoding_format: float)", 
            "Normalized embeddings",
            "Instruction formatting for Qwen3",
            "Caching support"
        ],
        "endpoints": ["/v1/embeddings", "/embeddings", "/v1/models", "/health"],
        "roo_code_compatible": True
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test connection to Ollama
        response = requests.get(f"{embedding_api.ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_available = any(model["name"] == embedding_api.model_name for model in models)
            
            return {
                "status": "healthy",
                "model_available": model_available,
                "model_name": embedding_api.model_name,
                "dimensions": embedding_api.dimensions,
                "timestamp": time.time()
            }
        else:
            return {"status": "unhealthy", "reason": "Ollama not responding"}
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings (OpenAI-compatible endpoint)"""
    # Normalize input to list
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input
    
    # Validate input
    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    
    # Check for empty strings
    if any(not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="Input cannot contain empty strings")
    
    if len(texts) > 100:  # Reasonable batch limit
        raise HTTPException(status_code=400, detail="Too many texts in batch (max 100)")
    
    # Validate encoding format
    if request.encoding_format not in ["float", "base64"]:
        raise HTTPException(
            status_code=400, 
            detail="encoding_format must be 'float' or 'base64'"
        )
    
    # Validate dimensions for MRL support
    if request.dimensions and request.dimensions not in MODEL_CONFIG["available_dimensions"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported dimension {request.dimensions}. "
                   f"Supported: {MODEL_CONFIG['available_dimensions']}"
        )
    
    # Generate embeddings with Qwen developer recommendations
    return embedding_api.generate_embeddings(
        texts=texts,
        encoding_format=request.encoding_format,
        task=request.task or "text_search",
        custom_instruction=request.instruction,
        target_dimensions=request.dimensions
    )

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings_legacy(request: EmbeddingRequest):
    """Legacy embeddings endpoint for compatibility"""
    return await create_embeddings(request)

@app.get("/v1/models")
async def list_models():
    """List available models with Qwen3 feature information"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_CONFIG["model_name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen",
                "permission": [],
                "root": MODEL_CONFIG["model_name"],
                "parent": None,
                "qwen_features": {
                    "instruction_aware": MODEL_CONFIG["supports_instructions"],
                    "mrl_support": MODEL_CONFIG["supports_mrl"],
                    "available_dimensions": MODEL_CONFIG["available_dimensions"],
                    "performance_improvement": MODEL_CONFIG["performance_improvement"],
                    "recommended_tasks": [
                        "text_search", "code_search", "document_retrieval",
                        "question_answering", "clustering", "classification"
                    ]
                }
            }
        ]
    }

if __name__ == "__main__":
    print(f"🚀 Starting Qwen3-Embedding-0.6B API server...")
    print(f"📊 Model: {MODEL_CONFIG['model_name']}")
    print(f"📏 Dimensions: {MODEL_CONFIG['dimensions']}")
    print(f"🎯 Use case: {MODEL_CONFIG['use_case']}")
    print(f"🔗 OpenAI-compatible endpoints:")
    print(f"   POST /v1/embeddings")
    print(f"   GET /v1/models")
    print(f"   GET /health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", timeout_keep_alive=60)
