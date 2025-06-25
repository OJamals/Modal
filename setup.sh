#!/bin/bash

# Qwen3 Embedding Setup Script
# Automates the complete setup process for RooCode integration

set -e  # Exit on any error

echo "🚀 Setting up Qwen3 Embedding for RooCode..."
echo "============================================="

# Check if required files exist
if [ ! -f "qwen3-embedding-0.6b.gguf" ]; then
    echo "❌ qwen3-embedding-0.6b.gguf not found!"
    echo "Please ensure the GGUF model file is in the current directory."
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Step 1: Setup Ollama model
echo "📦 Step 1: Setting up Ollama model..."
cat > Modelfile << EOF
FROM ./qwen3-embedding-0.6b.gguf
PARAMETER num_ctx 8192
PARAMETER embedding_only true
EOF

# Check if ollama is available
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found! Please install Ollama first:"
    echo "   curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# Create the model in Ollama
echo "Creating qwen3-embedding model in Ollama..."
ollama create qwen3-embedding -f Modelfile
echo "✅ Ollama model created successfully"

# Step 2: Install Python dependencies
echo "📦 Step 2: Installing Python dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Step 3: Setup Qdrant
echo "📦 Step 3: Setting up Qdrant vector database..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running! Please start Docker first."
    exit 1
fi

# Stop existing Qdrant container if it exists
if docker ps -a --format 'table {{.Names}}' | grep -q "qdrant"; then
    echo "Stopping existing Qdrant container..."
    docker stop qdrant || true
    docker rm qdrant || true
fi

# Start new Qdrant container
echo "Starting Qdrant container..."
docker run -d --name qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -e QDRANT__SERVICE__API_KEY="your-super-secret-qdrant-api-key" \
    -v "$(pwd)/qdrant_storage:/qdrant/storage" \
    qdrant/qdrant

# Wait for Qdrant to be ready
echo "Waiting for Qdrant to start..."
for i in {1..30}; do
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo "✅ Qdrant is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Qdrant failed to start"
        exit 1
    fi
    sleep 2
done

# Step 4: Start the API in background
echo "📦 Step 4: Starting OpenAI-compatible API..."
python qwen3-api.py &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API to start..."
for i in {1..20}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API is ready"
        break
    fi
    if [ $i -eq 20 ]; then
        echo "❌ API failed to start"
        kill $API_PID || true
        exit 1
    fi
    sleep 2
done

# Step 5: Setup Qdrant vector store
echo "📦 Step 5: Setting up Qdrant vector store..."
python qdrantsetup.py
echo "✅ Qdrant vector store configured"

# Step 6: Run verification tests
echo "📦 Step 6: Running verification tests..."
python test_setup.py

# Summary
echo ""
echo "🎉 Setup complete! Your Qwen3 embedding system is ready for Roo Code."
echo ""
echo "🔧 Roo Code Configuration:"
echo "   Embeddings Provider: OpenAI-compatible"
echo "   Base URL: http://localhost:8000"
echo "   API Key: your-super-secret-qdrant-api-key"
echo "   Model: qwen3"
echo "   Embedding Dimension: 1024"
echo "   Qdrant URL: http://localhost:6333"
echo "   Qdrant API Key: your-super-secret-qdrant-api-key"
echo "   Collection Name: qwen3_embedding"
echo ""
echo "🚀 Services running:"
echo "   - Qwen3 API: http://localhost:8000"
echo "   - Qdrant: http://localhost:6333"
echo "   - Ollama: http://localhost:11434"
echo ""
echo "💡 To stop the API: kill $API_PID"
echo "💡 To stop Qdrant: docker stop qdrant"
echo "💡 To restart: ./setup.sh"
