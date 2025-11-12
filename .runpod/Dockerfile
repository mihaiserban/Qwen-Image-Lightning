# Use a modern Runpod PyTorch base image
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

# Install dependencies including huggingface-cli
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors pillow runpod hf_transfer bitsandbytes "huggingface_hub[cli]"

# Set HuggingFace cache to RunPod's persistent storage
ENV HF_HOME=/runpod-volume/huggingface-cache
ENV HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/transformers

# Create cache directory
RUN mkdir -p /runpod-volume/huggingface-cache

# Download base model using modern hf download command (will use cache)
RUN hf download Qwen/Qwen-Image-Edit-2509 || echo "Model will be downloaded on first run from cache"

# Download Lightning LoRA weights (bf16 version) using modern hf download command (will use cache)
RUN hf download lightx2v/Qwen-Image-Lightning \
    --include "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors" || echo "LoRA will be downloaded on first run from cache"

# Copy handler file
WORKDIR /app
COPY handler.py .

# Set entrypoint
CMD ["python", "handler.py"]
