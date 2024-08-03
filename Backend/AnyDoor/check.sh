#!/bin/bash

url1="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth"
destination1="src/model_ckpts/dinov2_vitg14_pretrain.pth"

url2="https://huggingface.co/spaces/xichenhku/AnyDoor/resolve/main/epoch%3D1-step%3D8687.ckpt"
destination2="src/model_ckpts/epoch=1-step=8687.ckpt"

mkdir -p "src/model_ckpts"

wget -O "$destination1" "$url1"
wget -O "$destination2" "$url2"
