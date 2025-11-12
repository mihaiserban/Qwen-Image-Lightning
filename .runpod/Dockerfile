# Use a modern Runpod PyTorch base image
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

# Install dependencies including huggingface-cli
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors pillow runpod hf_transfer bitsandbytes "huggingface_hub[cli]"

# Download only specific files from the Hugging Face repos (avoid downloading the entire ~53GB)
RUN python - <<'PY'
from huggingface_hub import list_repo_files, hf_hub_download
import os

os.makedirs("/app/models", exist_ok=True)

# List files in the base model repo so you can see the exact filenames in build logs
print("Files in Qwen/Qwen-Image-Edit-2509:")
files = list_repo_files("Qwen/Qwen-Image-Edit-2509")
for f in files:
    print(f"  {f}")

# Replace these filenames with the exact file(s) you need from the printed list.
# Example placeholders — update to the real filenames from the repo listing.
base_model_filename = "model.safetensors"  # << replace with exact file from the list
lora_filename = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors"

# Download only the specified files into /app/models
print(f"Downloading {base_model_filename}...")
hf_hub_download(repo_id="Qwen/Qwen-Image-Edit-2509",
                filename=base_model_filename,
                local_dir="/app/models",
                local_dir_use_symlinks=False)

print(f"Downloading {lora_filename}...")
hf_hub_download(repo_id="lightx2v/Qwen-Image-Lightning",
                filename=lora_filename,
                local_dir="/app/models",
                local_dir_use_symlinks=False)

print("Downloaded requested files to /app/models")
PY

# Copy handler file
WORKDIR /app
COPY handler.py .

# Set entrypoint
CMD ["python", "handler.py"]
