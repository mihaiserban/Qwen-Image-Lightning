# Use a modern Runpod PyTorch base image
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

# Install dependencies including huggingface-cli
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors pillow runpod hf_transfer bitsandbytes "huggingface_hub[cli]"

# Download base model using huggingface-cli
RUN huggingface-cli download Qwen/Qwen-Image-Edit-2509

# Download Lightning LoRA weights using huggingface-cli
RUN huggingface-cli download lightx2v/Qwen-Image-Lightning \
    --include "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors"

# Copy handler file
WORKDIR /app
COPY handler.py .

# Set entrypoint
CMD ["python", "handler.py"]
