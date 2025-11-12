# Use RunPod base image
FROM runpod/base:0.6.3-cuda12.4.0

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

# Install dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install --upgrade -r /requirements.txt --no-cache-dir --system

# Pre-download models and LoRA weights to speed up cold starts
# Models will be cached in the container
RUN python3.11 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen-Image-Edit-2509', allow_patterns=['*.json', '*.safetensors', '*.model'])"
RUN python3.11 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='lightx2v/Qwen-Image-Lightning', filename='Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors')"

# Add handler file
ADD handler.py .

# Run the handler
CMD python -u /handler.py
