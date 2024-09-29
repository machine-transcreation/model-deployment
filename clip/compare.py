import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

base_dir = "./images"
original_dir = os.path.join(base_dir, "Original")
models_dirs = {
    "Platform": os.path.join(base_dir, "Platform"),
    "Google Imagen 3": os.path.join(base_dir, "Google Imagen"),
    "OpenAI Dalle 3": os.path.join(base_dir, "Dalle"),
    "Flux": os.path.join(base_dir, "Flux")
}

original_images = []
original_image_info = {}
for index, image_path in enumerate(os.listdir(original_dir)):
    img = Image.open(f'{original_dir}/{image_path}')
    original_images.append(img)
    original_image_info[index] = {"name": image_path.replace(".png", "").replace(".jpg", ""), "image": img}

all_images = [original_images]  
model_image_info = {}
for model_name, model_dir in models_dirs.items():
    model_images = []
    for image_path in os.listdir(model_dir):
        img = Image.open(f'{model_dir}/{image_path}')
        model_images.append(img)
    model_image_info[model_name] = model_images
    all_images.append(model_images)

flattened_images = original_images[:]
for model_images in model_image_info.values():
    flattened_images.extend(model_images)

inputs = processor(images=flattened_images, return_tensors="pt", padding=True)

with torch.no_grad():
    image_features = model.get_image_features(**inputs)

def compare_images(image_features, img1_idx, img2_idx, original_image, compared_image):
    cosine_sim = cosine_similarity(image_features[img1_idx].unsqueeze(0), image_features[img2_idx].unsqueeze(0))[0][0]
    ssim = structural_similarity(
        np.array(original_image.convert("L")), np.array(compared_image.convert("L"))
    )
    return cosine_sim, ssim

cosine_similarities = []
structural_similarities = []
labels = []

for index, (original_image_name, original_image) in original_image_info.items():
    for model_name, model_images in model_image_info.items():
        compared_image = model_images[index]
        compared_image_idx = len(original_images) + list(model_image_info.keys()).index(model_name) * len(original_images) + index
        
        cosine_sim, ssim = compare_images(image_features, index, compared_image_idx, original_image, compared_image)
        
        cosine_similarities.append(cosine_sim)
        structural_similarities.append(ssim)
        labels.append(f'{model_name} - {original_image_name}')

x = np.arange(len(labels)) 
width = 0.35  

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, cosine_similarities, width, label='Cosine Similarity')
rects2 = ax.bar(x + width/2, structural_similarities, width, label='Structural Similarity')

ax.set_xlabel('Model - Original Image')
ax.set_ylabel('Similarity Score')
ax.set_title('Cosine and SSIM Comparison for Model Outputs')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90)
ax.legend()

ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1.1, 0.1))

plt.tight_layout()
plt.show()
