import os
import pandas as pd
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor

base_dir = "images"
models = ["HILITE", "Google Imagen 3", "Dall-E"]
image_dir = {model: os.path.join(base_dir, model) for model in models}
original_dir = os.path.join(base_dir, "Original")

captions_df = pd.read_csv('captions.csv')

title_to_caption = {row['Title']: {model: row[model] for model in ["Original"] + models} for _, row in captions_df.iterrows()}

def load_image(filepath):
    return Image.open(filepath)

def load_images_and_captions(image_names, model_name):
    images = []
    captions = []
    for image_name in image_names:
        image_path = os.path.join(image_dir[model_name], image_name)
        image = load_image(image_path)
        images.append(image)
        captions.append(title_to_caption[image_name][model_name])
    return images, captions

device = "cpu"
clip_id = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_id)
text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)

class DirectionalSimilarity(torch.nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(device)}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity

dir_similarity_model = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)

results = {model: [] for model in models}

image_names = [img for img in os.listdir(original_dir) if img.endswith(".png")]

for image_name in image_names:
    image_name_key = image_name.replace(".png", "")
    
    original_image_path = os.path.join(original_dir, image_name)
    original_image = load_image(original_image_path)
    
    if image_name_key in title_to_caption:
        original_caption = title_to_caption.get(image_name_key, {}).get("Original", "")
        
        original_image_feat = dir_similarity_model.encode_image(original_image)

        for model in models:
            edited_image_path = os.path.join(image_dir[model], image_name)
            edited_image = load_image(edited_image_path)
            modified_caption = title_to_caption[image_name_key].get(model, "")

            edited_image_feat = dir_similarity_model.encode_image(edited_image)

            cosine_sim = F.cosine_similarity(original_image_feat, edited_image_feat).item()

            text_image_direction_similarity = dir_similarity_model(
                original_image, edited_image, original_caption, modified_caption
            )
            text_image_direction_similarity = float(text_image_direction_similarity.detach().cpu())

            results[model].append((cosine_sim, text_image_direction_similarity))
    else:
        print(f"Warning: '{image_name_key}' not found in captions CSV.")

plt.figure(figsize=(12, 8))

markers = ['o', 's', '^'] 
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

for idx, model in enumerate(models):
    cosine_sims = [res[0] for res in results[model]]
    text_img_sims = [res[1] for res in results[model]]
    
    combined = list(zip(cosine_sims, text_img_sims))
    
    sorted_combined = sorted(zip([res[0] for res in results[model]], [res[1] for res in results[model]]), key=lambda x: (-x[0], x[1]))
    sorted_cosine_sims, sorted_text_img_sims = zip(*sorted_combined)
    plt.plot(sorted_text_img_sims, sorted_cosine_sims, marker=markers[idx], label=model, linestyle='-', markersize=8, color=colors[idx], alpha=0.7)

plt.title('Trade-off: CLIP Score vs Text-Image Direction Similarity', fontsize = 16)
plt.xlabel('Text-Image Direction Similarity', fontsize = 14)
plt.ylabel('CLIP Score (Cosine Similarity of Images)', fontsize = 14)
plt.legend(title = 'Diffusion Models', title_fontsize = 12, fontsize = 10)
plt.grid(True, linestyle = '--', alpha = 0.7)
plt.tight_layout()

plt.show()