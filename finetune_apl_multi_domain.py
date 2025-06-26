# fine_tune_multidomain_attention_pooling.py
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from typing import List, Dict
import random
from clip_test import RoadSignClassifier
from prompts import prompts_india, prompts_china, prompts_germany

# Configuration
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-5
WEIGHT_DECAY = 0.01
SAVE_PATH = "finetuned_multidomain_clip.pth"

# Contrastive Loss
def clip_contrastive_loss(image_features, text_features, temperature=1.0):
    image_features = nn.functional.normalize(image_features, dim=-1)
    text_features = nn.functional.normalize(text_features, dim=-1)
    logits = image_features @ text_features.T / temperature
    labels = torch.arange(len(image_features)).to(image_features.device)
    return nn.functional.cross_entropy(logits, labels)

# Multi-domain Dataset
class MultiDomainRoadSignDataset(Dataset):
    def __init__(self, image_dirs: Dict[str, str], categories: List[str], prompt_variations: Dict[str, List[List[str]]]):
        self.categories = [c.lower() for c in categories]
        self.prompt_variations = prompt_variations
        self.samples = []

        for domain, image_dir in image_dirs.items():
            for fname in os.listdir(image_dir):
                if not fname.lower().endswith(('png', 'jpg', 'jpeg')):
                    continue
                category = ' '.join(fname.split('_')[:-1]).lower()
                if category not in self.categories:
                    continue
                category_idx = self.categories.index(category)
                self.samples.append({
                    'image_path': os.path.join(image_dir, fname),
                    'category_idx': category_idx
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert("RGB")
        return image, sample['category_idx']

# Load and prepare model
def prepare_model():
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.vision_model.named_parameters():
        if 'encoder.layers.11' in name or 'post_layernorm' in name:
            param.requires_grad = True
    return model

# Training loop with multi-domain prompts
def train_model(model, dataloader, processor, prompt_variations, categories):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for images, category_indices in dataloader:
            images = list(images)
            category_indices = category_indices.tolist()

            prompts = [random.choice(prompt_variations['india'][i]) for i in category_indices]
            #prompts = [random.choice(prompt_variations['china'][i]) for i in category_indices]
            #prompts = [random.choice(prompt_variations['germany'][i]) for i in category_indices]

            text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            image_inputs = processor(images=images, return_tensors="pt").to(DEVICE)

            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)

            loss = clip_contrastive_loss(image_features, text_features, temperature=0.07)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

# Prepare data
def load_training_data():
    classifier = RoadSignClassifier(use_finetuned=False)
    categories = classifier.categories

    image_dirs = {
        'india': "train_data_India_5"
        #'china': "train_data_China_5"
        #'germany': "train_data_Germany_5"
    }

    prompt_variations = {
        'india': prompts_india
        #'china': prompts_china
        #'germany': prompts_germany
    }

    dataset = MultiDomainRoadSignDataset(image_dirs, categories, prompt_variations)

    def collate_fn(batch):
        images, category_indices = zip(*batch)
        return list(images), torch.tensor(category_indices)

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn), classifier.processor, prompt_variations, categories

if __name__ == "__main__":
    dataloader, processor, prompt_variations, categories = load_training_data()
    model = prepare_model()
    train_model(model, dataloader, processor, prompt_variations, categories)
