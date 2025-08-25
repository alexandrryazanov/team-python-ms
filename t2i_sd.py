from diffusers import StableDiffusionPipeline
import torch

import os
os.environ["HF_HOME"] = "./hf_home"
os.environ["HF_CACHE_DIR"] = "./cache"

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    cache_dir='./cache',
    )
pipe = pipe.to("mps")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  

image.save("astronaut_rides_horse.png")
