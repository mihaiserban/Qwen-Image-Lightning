import runpod
from diffusers import DiffusionPipeline
import torch
from io import BytesIO
import base64
from PIL import Image
import numpy as np

# Load model on startup (module level)
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

# Load Lightning LoRA (8-step V2.0)
lora_path = "/workspace/models/Qwen-Image-Lightning-8steps-V2.0.safetensors"
pipe.load_lora_weights(lora_path)

# Optimize pipeline
pipe.enable_xformers_memory_efficient_attention()
pipe.unet.to(memory_format=torch.channels_last)


def add_film_grain(img: Image.Image, intensity: float = 0.05) -> Image.Image:
    """Add authentic film grain."""
    arr = np.array(img, dtype=np.float32) / 255.0
    noise = np.random.normal(0, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def handler(event):
    """
    RunPod handler function. Receives job input and returns output.
    """
    try:
        input_data = event["input"]
        
        # Validate required inputs
        if "selfie" not in input_data:
            return {"error": "Missing required field: selfie"}
        
        # Get input parameters
        selfie_b64 = input_data["selfie"]
        prompt = input_data.get("prompt", "")
        neg_prompt = input_data.get("negative_prompt", "blurry, deformed, ugly, extra limbs, watermark, cartoon, lowres")
        steps = input_data.get("num_inference_steps", 8)
        guidance = input_data.get("guidance_scale", 1.0)
        strength = input_data.get("strength", 0.7)
        seed = input_data.get("seed", 42)
        
        # Decode base64 image
        try:
            selfie = Image.open(BytesIO(base64.b64decode(selfie_b64))).convert("RGB")
            selfie = selfie.resize((1024, 1024), Image.LANCZOS)
        except Exception as e:
            return {"error": f"Invalid image data: {str(e)}"}
        
        # Generate image
        try:
            generator = torch.Generator("cuda").manual_seed(seed)
            output = pipe(
                image=selfie,
                prompt=prompt,
                negative_prompt=neg_prompt,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]
        except torch.cuda.OutOfMemoryError:
            return {"error": "GPU out of memory. Try reducing image size or steps."}
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}
        
        # Post-process: 9:16 aspect ratio, film grain, warm tone
        w, h = output.size
        target_h = int(w * 16 / 9)
        output = output.resize((w, target_h), Image.LANCZOS)
        output = add_film_grain(output, intensity=0.05)
        
        # Warm tone enhancement
        from PIL import ImageEnhance
        output = ImageEnhance.Color(output).enhance(1.08)
        
        # Encode result to base64
        buffered = BytesIO()
        output.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"image": img_str, "prompt": prompt}
        
    except Exception as e:
        return {"error": str(e)}


# Required by RunPod
runpod.serverless.start({"handler": handler})
