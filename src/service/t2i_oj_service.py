from flask import Flask, send_file, request
from diffusers import StableDiffusionPipeline
from io import BytesIO
# from PIL import Image
import torch
import os

os.environ["HF_HOME"] = "./hf_home"
os.environ["HF_CACHE_DIR"] = "./cache"

model_id = "prompthero/openjourney"
pipe_oj = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    cache_dir='./cache')
pipe_oj = pipe_oj.to("mps")

app = Flask(__name__)

@app.route('/image', methods=['GET'])
def hello_world():
    print("new request received for /hello...")

    prompt = request.args.get('promt')
    if prompt is None:
        prompt = "backend developer works very hard all day"
    print("prompt is:", prompt)

    img = pipe_oj(prompt=prompt).images[0]
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5556)