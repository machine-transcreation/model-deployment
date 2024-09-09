from diffusers import StableDiffusionXLControlNetXSPipeline, ControlNetXSAdapter, AutoencoderKL
import numpy as np
import torch

import cv2
from PIL import Image
import requests
from io import BytesIO
import base64
import torch
import runpod

negative_prompt = "low quality, bad quality, sketches"

controlnet_conditioning_scale = 0.5

vae = AutoencoderKL.from_pretrained("./sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")

controlnet = ControlNetXSAdapter.from_pretrained(
    "./Testing-ConrolNetXS-SDXL-canny", torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusionXLControlNetXSPipeline.from_pretrained(
    "./stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

pipe.enable_model_cpu_offload()

def load_image_from_base64(base64_str: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image

def load_image(image_file: str) -> Image.Image:
    if image_file.startswith('https://') or image_file.startswith('http://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = load_image_from_base64(image_file)
    return image


def get_canny_image(image: Image, low_thresh: int, high_thresh: int):
    image = np.array(image)
    image = cv2.Canny(image, low_thresh, high_thresh)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    return canny_image

def run_local(image: Image, prompt: str, canny_low: int, canny_high: int, cfg: float, steps: int):
    image = pipe(
        prompt = prompt, 
        controlnet_conditioning_scale = controlnet_conditioning_scale, 
        image = get_canny_image(image, canny_low, canny_high),
        guidance_scale = cfg,
        num_inference_steps = steps,
        target_size = image.size
    ).images[0]

    bytes = BytesIO()
    image.save(bytes, format = "PNG")

    return base64.b64encode(bytes.getvalue()).decode("utf-8")

def handler(job):
    input = job["input"]

    image = load_image(input["image"])
    prompt = input["prompt"]
    guidance_scale = float(input["guidance_scale"])
    canny_low = int(input["canny_low"])
    canny_high = int(input["canny_high"])
    steps = int(input["steps"])

    return run_local(image, prompt, canny_low, canny_high, guidance_scale, steps)

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})