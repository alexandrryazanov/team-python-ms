from diffusers import StableDiffusionPipeline
import torch

import os
os.environ["HF_HOME"] = "./hf_home"
os.environ["HF_CACHE_DIR"] = "./cache"

model_id = "prompthero/openjourney"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    cache_dir='./cache')
pipe = pipe.to("mps")

prompt = "An astronaut riding a green horse"

image = pipe(prompt=prompt).images[0]
image.save("./img_journey.png")