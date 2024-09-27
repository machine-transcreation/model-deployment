FROM python:3.12.4-slim-bookworm

WORKDIR /

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | bash
RUN yum install git-lfs -y
RUN git lfs install

WORKDIR /src/
RUN git clone https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/
RUN git clone https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/
RUN git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/

COPY builder/requirements.txt .

RUN pip install --no-cache-dir --progress-bar off -r requirements.txt --no-use-pep517


ADD src/controlnet-canny-sdxl-1.0 .
ADD src/sdxl-vae-fp16-fix .
ADD src/stable-diffusion-xl-base-1.0/text_encoder_2 .
ADD src/stable-diffusion-xl-base-1.0/text_encoder .
ADD src/stable-diffusion-xl-base-1.0/tokenizer_2 .
ADD src/stable-diffusion-xl-base-1.0/vae .
ADD src/stable-diffusion-xl-base-1.0/scheduler .
ADD src/stable-diffusion-xl-base-1.0/vae_decoder .
ADD src/stable-diffusion-xl-base-1.0/vae_1_0 .
ADD src/stable-diffusion-xl-base-1.0/tokenizer .
ADD src/stable-diffusion-xl-base-1.0/vae_encoder .
ADD src/stable-diffusion-xl-base-1.0/.gitattributes .
ADD src/stable-diffusion-xl-base-1.0/01.png .
ADD src/stable-diffusion-xl-base-1.0/comparison.png .
ADD src/stable-diffusion-xl-base-1.0/model_index.json .
ADD src/stable-diffusion-xl-base-1.0/pipeline.png .
ADD src/stable-diffusion-xl-base-1.0/README.md .
ADD src/stable-diffusion-xl-base-1.0/sd_xl_base_1.0_0.9vae.safetensors .
ADD src/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors .
ADD src/stable-diffusion-xl-base-1.0/sd_xl_offset_example-lora_1.0.safetensors .
ADD src/stable-diffusion-xl-base-1.0/unet/config.json .
ADD src/stable-diffusion-xl-base-1.0/unet/diffusion_flax_model.msgpack .
ADD src/stable-diffusion-xl-base-1.0/unet/diffusion_pytorch_model.fp16.safetensors .
ADD src/stable-diffusion-xl-base-1.0/unet/diffusion_pytorch_model.safetensors .
ADD src/stable-diffusion-xl-base-1.0/unet/model.onnx .
ADD src/stable-diffusion-xl-base-1.0/unet/model.onnx_data .
ADD src/stable-diffusion-xl-base-1.0/unet/openvino_model.bin .
ADD src/stable-diffusion-xl-base-1.0/unet/openvino_model.xml .

CMD ["python", "-u", "/handler.py"]