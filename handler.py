import runpod
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.models import QwenImageTransformer2DModel
import torch
import math
from io import BytesIO
import base64
from PIL import Image

# Load model with Lightning LoRA on startup
model = QwenImageTransformer2DModel.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", subfolder="transformer", torch_dtype=torch.bfloat16
)

# Configure scheduler for Lightning (shift=3 for distillation)
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=model,
    scheduler=scheduler,
    torch_dtype=torch.bfloat16
)

# Load Lightning LoRA weights (8-step bf16 version for consistency with torch.bfloat16)
pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", 
                       weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors")

pipe = pipe.to("cuda")

def decode_base64_to_image(base64_string):
    """
    Decode base64 string to PIL Image.
    """
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))

def handler(event):
    """
    Runpod handler function. Receives job input and returns output.
    Supports:
    - Single image mode: image_base64 + prompt
    - Dual image mode: image1_base64 + image2_base64 + prompt
    Optional parameters:
    - num_inference_steps (default: 8 for Lightning)
    - true_cfg_scale (default: 1.0 for Lightning)
    """
    try:
        input_data = event["input"]
        prompt = input_data.get("prompt", "Enhance the image")
        
        # Get optional parameters with Lightning defaults
        num_inference_steps = input_data.get("num_inference_steps", 8)
        true_cfg_scale = input_data.get("true_cfg_scale", 1.0)
        
        # Check for single image mode
        image_base64 = input_data.get("image_base64")
        
        # Check for dual image mode
        image1_base64 = input_data.get("image1_base64")
        image2_base64 = input_data.get("image2_base64")
        
        # Prepare pipeline arguments
        pipe_args = {
            "prompt": prompt,
            "generator": torch.Generator(device="cuda").manual_seed(42),
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": " ",
            "num_inference_steps": num_inference_steps,
        }
        
        # Determine mode and process images
        if image_base64:
            # Single image mode
            input_image = decode_base64_to_image(image_base64)
            pipe_args["image"] = input_image
        elif image1_base64 and image2_base64:
            # Dual image mode
            image1 = decode_base64_to_image(image1_base64)
            image2 = decode_base64_to_image(image2_base64)
            pipe_args["image"] = [image1, image2]
        else:
            return {"error": "Missing image parameters. Provide either 'image_base64' or both 'image1_base64' and 'image2_base64'."}
        
        output_image = pipe(**pipe_args).images[0]
        
        # Convert output to base64
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"output_image_base64": img_str, "prompt": prompt}
    except Exception as e:
        return {"error": str(e)}

# Required by Runpod
runpod.serverless.start({"handler": handler})
