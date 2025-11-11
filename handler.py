import runpod
from diffusers import QwenImageEditPlusPipeline
import torch
from io import BytesIO
import base64
from PIL import Image

# Load model on startup
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.float16
).to("cuda")

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
    """
    try:
        input_data = event["input"]
        prompt = input_data.get("prompt", "Enhance the image")
        
        # Check for single image mode
        image_base64 = input_data.get("image_base64")
        
        # Check for dual image mode
        image1_base64 = input_data.get("image1_base64")
        image2_base64 = input_data.get("image2_base64")
        
        # Determine mode and process images
        if image_base64:
            # Single image mode
            input_image = decode_base64_to_image(image_base64)
            output_image = pipe(image=input_image, prompt=prompt).images[0]
        elif image1_base64 and image2_base64:
            # Dual image mode
            image1 = decode_base64_to_image(image1_base64)
            image2 = decode_base64_to_image(image2_base64)
            output_image = pipe(image=[image1, image2], prompt=prompt).images[0]
        else:
            return {"error": "Missing image parameters. Provide either 'image_base64' or both 'image1_base64' and 'image2_base64'."}
        
        # Convert output to base64
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"output_image_base64": img_str, "prompt": prompt}
    except Exception as e:
        return {"error": str(e)}

# Required by Runpod
runpod.serverless.start({"handler": handler})
