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

image_dir = "./images"
images = []
info = {}

for index, image_path in enumerate(os.listdir(image_dir)):
    img = Image.open(f'{image_dir}/{image_path}')
    images.append(img)
    info[index] = {"name": image_path.replace(".png", "").replace(".jpg", ""), "image": img}

inputs = processor(images=images, return_tensors="pt", padding=True)

with torch.no_grad():
    image_features = model.get_image_features(**inputs)

cosine_sim_1 = cosine_similarity(image_features[0].unsqueeze(0), image_features[1].unsqueeze(0))[0][0]
cosine_sim_2 = cosine_similarity(image_features[0].unsqueeze(0), image_features[2].unsqueeze(0))[0][0]

ssim_1 = structural_similarity(np.array(info[0]["image"].convert("L")), np.array(info[1]["image"].convert("L")))
ssim_2 = structural_similarity(np.array(info[0]["image"].convert("L")), np.array(info[2]["image"].convert("L")))

labels = [f'{info[1]["name"]}', f'{info[2]["name"]}']
cosine_similarities = [cosine_sim_1, cosine_sim_2]
structural_similarities = [ssim_1, ssim_2]

x = np.arange(len(labels)) 
width = 0.35  

fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, cosine_similarities, width, label='Cosine Similarity')
rects2 = ax.bar(x + width/2, structural_similarities, width, label='Structural Similarity')

ax.set_xlabel('Images')
ax.set_ylabel('Similarity Score')
ax.set_title('Cosine and SSIM')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.set_ylim(min(min(cosine_similarities), min(structural_similarities)) - 0.1, 1.1)

# Show the plot
plt.tight_layout()
plt.show()
