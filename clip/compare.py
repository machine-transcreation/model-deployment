import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_ssim(image1, image2):
    image1 = image1.resize(image2.size)
    image1_gray = np.array(image1.convert('L'))
    image2_gray = np.array(image2.convert('L'))
    return ssim(image1_gray, image2_gray)

original_folder = 'images/Original/'
HILITE_folder = 'images/HILITE/'
google_folder = 'images/Google Imagen 3/'
dalle_folder = 'images/Dall-E/'

original_images = sorted(os.listdir(original_folder))

cosine_sims = {'HILITE': [], 'Google Imagen 3': [], 'Dall-E': []}
ssims = {'HILITE': [], 'Google Imagen 3': [], 'Dall-E': []}

prompts = {
    "black bear": "make this bear a black bear",
    "changing bear": "change black bear to white bear",
    "chinese school": "Make this school extremely and obviously Chinese",
    "duck-swan": "change the duck to a swan",
    "indian dal": "replace the ramen with dal (lentil soup)",
    "jaguar tiger": "change the jaguar into a tiger",
    "matcha-chai": "change the matcha into a cup of chai",
    "roti": "change pancake to roti or indian food",
    "tasty panner": "change the churrasco on the grill to paneer"
}

for image_name in original_images:
    original_image_path = os.path.join(original_folder, image_name)
    original_image = Image.open(original_image_path)

    for folder_name, folder_path in zip(['HILITE', 'Google Imagen 3', 'Dall-E'], 
                                        [HILITE_folder, google_folder, dalle_folder]):
        comparison_image_path = os.path.join(folder_path, image_name)
        comparison_image = Image.open(comparison_image_path)
        
        inputs_original = processor(images=original_image, return_tensors="pt", padding=True)
        inputs_comparison = processor(images=comparison_image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            embedding_original = model.get_image_features(**inputs_original)
            embedding_comparison = model.get_image_features(**inputs_comparison)
        
        cos_sim = torch.nn.functional.cosine_similarity(embedding_original, embedding_comparison).item()
        cosine_sims[folder_name].append(cos_sim)
        
        ssim_value = compute_ssim(original_image, comparison_image)
        ssims[folder_name].append(ssim_value)

avg_cosine_sims = {key: np.mean(values) for key, values in cosine_sims.items()}
avg_ssims = {key: np.mean(values) for key, values in ssims.items()}

categories = list(avg_cosine_sims.keys())
avg_cosine = list(avg_cosine_sims.values())
avg_ssim = list(avg_ssims.values())

fig, ax = plt.subplots(figsize=(10, 6))

width = 0.35
x = np.arange(len(categories))

rects1 = ax.bar(x - width/2, avg_cosine, width, label='Cosine Similarity', color='#1f77b4')
rects2 = ax.bar(x + width/2, avg_ssim, width, label='Structural Similarity', color='#ff7f0e')

ax.set_ylabel('Average Similarity Score')
ax.set_title('Average Similarity Scores for Image Edits')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
            xy = (rect.get_x() + rect.get_width() / 2, height),
            xytext = (0, 3), 
            textcoords = "offset points",
            ha = 'center', va = 'bottom'
        )

autolabel(rects1)
autolabel(rects2)
plt.yticks(np.arange(0, 1, step=0.05))

fig.tight_layout()
plt.savefig('results.png') 
plt.show()