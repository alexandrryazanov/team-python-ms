# Text to image generation

## Startup command

(python3 should be installed on yout computer)

./startup.sh

## Example

### Run openjournye model

./venv/bin/python3 ./src/t2i_generators/t2i_oj.py

### Run stable-diffusion-v1-5 model

./venv/bin/python3 ./src/t2i_generators/t2i_oj.py

### Run local service that generates images based on openjournye model
./venv/bin/python3 ./src/service/t2i_oj_service.py
