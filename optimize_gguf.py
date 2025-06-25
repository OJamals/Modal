#!/usr/bin/env python3
"""
GGUF Model Optimizer for Qwen3-Embedding-0.6B

This script optimizes GGUF models downloaded through Ollama by:
1. Locating the model in Ollama's blob storage
2. Copying it to the local directory with a standardized name
3. Creating an optimized Modelfile for embedding-only usage
4. Registering the optimized model with Ollama

Supports multiple quantization levels (Q8_0, Q4_K_M, etc.)
"""

import os
import sys
import json
import shutil
import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Dict, List

class GGUFOptimizer:
    def __init__(self):
        self.ollama_models_dir = Path.home() / ".ollama" / "models"
        self.blobs_dir = self.ollama_models_dir / "blobs"
        self.manifests_dir = self.ollama_models_dir / "manifests"
        
    def find_qwen3_models(self) -> List[Dict]:
        """Find all Qwen3 embedding models in Ollama."""
        models = []
        
        # Get list of models from Ollama
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            for line in result.stdout.split('\n'):
                if 'qwen3' in line.lower() and 'embedding' in line.lower():
                    parts = line.split()
                    if parts:
                        models.append({
                            'name': parts[0],
                            'size': parts[1] if len(parts) > 1 else 'Unknown'
                        })
        except subprocess.CalledProcessError as e:
            print(f"Error getting Ollama models: {e}")
            
        return models
    
    def get_model_manifest(self, model_name: str) -> Optional[Dict]:
        """Get the manifest for a specific model."""
        # Convert model name to manifest path
        manifest_path = self.manifests_dir / "registry.ollama.ai" / model_name.replace(':', '/')
        
        if not manifest_path.exists():
            # Try different path formats
            alt_paths = [
                self.manifests_dir / model_name.replace(':', '/'),
                self.manifests_dir / "huggingface.co" / model_name.replace(':', '/'),
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    manifest_path = alt_path
                    break
            else:
                return None
        
        try:
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def find_gguf_blob(self, model_name: str) -> Optional[Path]:
        """Find the GGUF blob file for a model."""
        manifest = self.get_model_manifest(model_name)
        if not manifest:
            print(f"Could not find manifest for {model_name}")
            return None
        
        # Look for GGUF files in the layers
        for layer in manifest.get('layers', []):
            if layer.get('mediaType') == 'application/vnd.ollama.image.model':
                blob_hash = layer.get('digest', '').replace('sha256:', '')
                if blob_hash:
                    blob_path = self.blobs_dir / f"sha256-{blob_hash}"
                    if blob_path.exists():
                        return blob_path
        
        return None
    
    def verify_gguf_file(self, file_path: Path) -> Dict:
        """Verify and get information about a GGUF file."""
        if not file_path.exists():
            return {'valid': False, 'error': 'File does not exist'}
        
        try:
            # Read first 8 bytes to check GGUF magic
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic != b'GGUF':
                    return {'valid': False, 'error': 'Not a valid GGUF file'}
                
                # Read version
                version = int.from_bytes(f.read(4), byteorder='little')
                
            # Get file size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            return {
                'valid': True,
                'version': version,
                'size_mb': round(size_mb, 1),
                'path': str(file_path)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def copy_and_optimize_model(self, source_path: Path, target_name: str) -> bool:
        """Copy GGUF model and create optimized Modelfile."""
        target_path = Path(f"{target_name}.gguf")
        
        # Copy the GGUF file
        print(f"Copying {source_path} to {target_path}...")
        try:
            shutil.copy2(source_path, target_path)
            print(f"✅ Model copied to {target_path}")
        except Exception as e:
            print(f"❌ Error copying file: {e}")
            return False
        
        # Create optimized Modelfile
        modelfile_content = f"""FROM ./{target_name}.gguf

# Embedding-specific optimizations
PARAMETER num_ctx 8192
PARAMETER embedding_only true
PARAMETER num_thread {os.cpu_count() or 4}
PARAMETER use_mmap true
PARAMETER use_mlock false

# Performance tuning
PARAMETER repeat_penalty 1.0
PARAMETER temperature 0.0
PARAMETER top_p 1.0
"""
        
        modelfile_path = Path("Modelfile")
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        print(f"✅ Created optimized Modelfile")
        
        # Register with Ollama
        try:
            subprocess.run(
                ["ollama", "create", target_name, "-f", "Modelfile"],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✅ Registered optimized model as '{target_name}' in Ollama")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error registering model with Ollama: {e}")
            return False
    
    def optimize_model(self, model_name: str, output_name: str = "qwen3-embedding") -> bool:
        """Main optimization workflow."""
        print(f"🔍 Optimizing model: {model_name}")
        print("=" * 50)
        
        # Find the GGUF blob
        blob_path = self.find_gguf_blob(model_name)
        if not blob_path:
            print(f"❌ Could not find GGUF blob for {model_name}")
            return False
        
        # Verify the GGUF file
        info = self.verify_gguf_file(blob_path)
        if not info['valid']:
            print(f"❌ Invalid GGUF file: {info['error']}")
            return False
        
        print(f"📁 Found GGUF file: {blob_path}")
        print(f"📊 Size: {info['size_mb']} MB")
        print(f"📄 GGUF Version: {info['version']}")
        
        # Copy and optimize
        return self.copy_and_optimize_model(blob_path, output_name)

def main():
    """Main entry point."""
    optimizer = GGUFOptimizer()
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        output_name = sys.argv[2] if len(sys.argv) > 2 else "qwen3-embedding"
    else:
        # Interactive mode - show available models
        print("🔍 Finding Qwen3 embedding models in Ollama...")
        models = optimizer.find_qwen3_models()
        
        if not models:
            print("❌ No Qwen3 embedding models found in Ollama.")
            print("\nTo download a model, try:")
            print("  ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")
            print("  ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q4_K_M")
            return 1
        
        print("\n📋 Available Qwen3 embedding models:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model['name']} ({model['size']})")
        
        # Get user selection
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            model_idx = int(choice) - 1
            if model_idx < 0 or model_idx >= len(models):
                raise ValueError("Invalid selection")
            model_name = models[model_idx]['name']
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid selection or cancelled.")
            return 1
        
        output_name = input("Output name (default: qwen3-embedding): ").strip() or "qwen3-embedding"
    
    # Run optimization
    success = optimizer.optimize_model(model_name, output_name)
    
    if success:
        print("\n🎉 Optimization complete!")
        print(f"\n📝 Next steps:")
        print(f"1. Start the API: python qwen3-api.py")
        print(f"2. Setup Qdrant: python qdrantsetup.py")
        print(f"3. Test everything: python test_setup.py")
        print(f"\n🔧 Model name for API: {output_name}")
        return 0
    else:
        print("\n❌ Optimization failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
