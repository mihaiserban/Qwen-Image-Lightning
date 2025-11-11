FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

# Accept HF_TOKEN as build argument
ARG HF_TOKEN

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Download model with authentication token
RUN huggingface-cli download lightx2v/Qwen-Image-Lightning \
    Qwen-Image-Lightning-8steps-V2.0.safetensors --local-dir /workspace/models \
    --token $HF_TOKEN

CMD ["python", "handler.py"]
