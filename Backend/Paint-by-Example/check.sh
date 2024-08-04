#!/bin/bash

url1="https://huggingface.co/Fantasy-Studio/Paint-by-Example/resolve/main/model.ckpt"
destination1="/src/checkpoints/model.ckpt"

mkdir -p "src/checkpoints"

wget -O "$destination1" "$url1"
