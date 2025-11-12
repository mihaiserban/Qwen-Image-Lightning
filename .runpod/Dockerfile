# Use a modern Runpod PyTorch base image
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

# Install dependencies including huggingface-cli
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors pillow runpod hf_transfer bitsandbytes "huggingface_hub[cli]"

# Set HuggingFace cache to use workspace which RunPod mounts
ENV HF_HOME=/workspace/huggingface-cache
ENV HF_HUB_CACHE=/workspace/huggingface-cache/hub
ENV TRANSFORMERS_CACHE=/workspace/huggingface-cache/transformers

# Pre-download models during build (optional - they'll be cached in workspace at runtime)
# Comment out these lines if you prefer to download on first run
RUN hf download Qwen/Qwen-Image-Edit-2509 || true
RUN hf download lightx2v/Qwen-Image-Lightning \
    --include "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors" || true

# Copy handler file
WORKDIR /app
COPY handler.py .

# Set entrypoint
CMD ["python", "handler.py"]
