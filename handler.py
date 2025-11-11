# handler.py — RunPod Serverless
# Model: Qwen-Image-Edit-Lightning-8steps-V2.0
# Input: base64 selfie + prompt
# Output: base64 PNG (9:16, film grain, warm tone)

import base64
import io
import torch
import numpy as np
from PIL import Image, ImageEnhance
from diffusers import DiffusionPipeline
from safetensors.torch import load_file

# Global pipeline (loaded once per worker)
pipe = None

def init_pipeline():
    global pipe
    if pipe is not None:
        return pipe

    # Load base Qwen-Image-Edit (from HF)
    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")

    # Load Lightning LoRA (8-step V2.0 — best quality/speed)
    lora_path = "/workspace/models/Qwen-Image-Lightning-8steps-V2.0.safetensors"
    pipe.load_lora_weights(lora_path)

    # Optimize
    pipe.enable_xformers_memory_efficient_attention()
    pipe.unet.to(memory_format=torch.channels_last)
    return pipe

def add_film_grain(img: Image.Image, intensity: float = 0.05) -> Image.Image:
    """Add authentic film grain."""
    arr = np.array(img, dtype=np.float32) / 255.0
    noise = np.random.normal(0, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def handler(job):
    global pipe
    input_data = job["input"]

    # --- Input ---
    selfie_b64 = input_data["selfie"]
    prompt = input_data.get("prompt", "")
    neg_prompt = input_data.get("negative_prompt", "blurry, deformed, ugly, extra limbs, watermark, cartoon, lowres")
    steps = input_data.get("num_inference_steps", 8)
    guidance = input_data.get("guidance_scale", 1.0)
    strength = input_data.get("strength", 0.7)
    seed = input_data.get("seed", 42)

    # --- Decode selfie ---
    selfie = Image.open(io.BytesIO(base64.b64decode(selfie_b64))).convert("RGB")
    selfie = selfie.resize((1024, 1024), Image.LANCZOS)

    # --- Initialize pipeline ---
    pipe = init_pipeline()

    # --- Generate / Edit ---
    generator = torch.Generator("cuda").manual_seed(seed)
    output = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=selfie,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator
    ).images[0]

    # --- Post-process: 9:16, grain, warm tone ---
    w, h = output.size
    target_h = int(w * 16 / 9)
    output = output.resize((w, target_h), Image.LANCZOS)
    output = add_film_grain(output, intensity=0.05)
    output = ImageEnhance.Color(output).enhance(1.08)  # Warm afternoon light

    # --- Encode result ---
    buf = io.BytesIO()
    output.save(buf, format="PNG", optimize=True)
    result_b64 = base64.b64encode(buf.getvalue()).decode()

    return {"image": result_b64}

# Local test
if __name__ == "__main__":
    import json, sys
    job = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    print(json.dumps(handler(job)))