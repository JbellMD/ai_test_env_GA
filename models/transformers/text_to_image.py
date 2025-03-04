from diffusers import StableDiffusionPipeline
import torch

class TextToImageModel:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32
        ).to(device)
    
    def generate(self, prompt, num_images=1, guidance_scale=7.5, num_inference_steps=50):
        with torch.autocast(self.device):
            images = self.pipe(
                [prompt] * num_images,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images
        return images