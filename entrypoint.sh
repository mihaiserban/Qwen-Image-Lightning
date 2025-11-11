#!/bin/bash
set -e
mkdir -p /workspace/models
cd /workspace/models
if [ ! -f /workspace/.deps_installed ]; then
  pip install -r /app/requirements.txt
  touch /workspace/.deps_installed
fi
if [ ! -f Qwen-Image-Lightning-8steps-V2.0.safetensors ]; then
  huggingface-cli download lightx2v/Qwen-Image-Lightning Qwen-Image-Lightning-8steps-V2.0.safetensors --local-dir .
fi
# Add other models as needed (e.g., Edit Lightning)
exec "$@"