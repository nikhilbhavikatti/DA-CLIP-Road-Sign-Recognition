import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from typing import List
import random
from clip_test import RoadSignClassifier

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-5
WEIGHT_DECAY = 0.01
SAVE_PATH = "finetuned_da_clip.pth"

# Contrastive Loss
def clip_contrastive_loss(image_features, text_features, temperature=1.0):
    image_features = nn.functional.normalize(image_features, dim=-1)
    text_features = nn.functional.normalize(text_features, dim=-1)

    logits = image_features @ text_features.T / temperature
    labels = torch.arange(len(image_features)).to(image_features.device)
    return nn.functional.cross_entropy(logits, labels)

class RoadSignDataset(Dataset):
    def __init__(self, image_dir: str, processor, categories: List[str], prompt_variations: List[List[str]]):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.processor = processor
        self.categories = [c.lower() for c in categories]
        self.prompt_variations = prompt_variations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Extract category from filename
        base_name = os.path.basename(image_path)
        category = ' '.join(base_name.split('_')[:-1]).lower()
        if category not in self.categories:
            raise ValueError(f"Unknown category '{category}' found in file: {base_name}")

        category_idx = self.categories.index(category)
        prompt = random.choice(self.prompt_variations[category_idx])

        return image, prompt

# Load model and unfreeze attention pooling
def prepare_model():
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.vision_model.named_parameters():
        if 'encoder.layers.11' in name or 'post_layernorm' in name:
            param.requires_grad = True
    return model

# Training loop
def train_model(model, dataloader, processor):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LR, weight_decay=WEIGHT_DECAY)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, prompts in dataloader:
            image_inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)

            loss = clip_contrastive_loss(image_features, text_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

# Load dataset and prompts
def load_training_data():

    image_dir = "train_data_Germany_1"
    classifier = RoadSignClassifier(use_finetuned=False)

    dataset = RoadSignDataset(
        image_dir=image_dir,
        processor=classifier.processor,
        categories=classifier.categories,
        prompt_variations=classifier.prompt_variations
    )

    def collate_fn(batch):
        images, prompts = zip(*batch)
        return list(images), list(prompts)

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn), classifier.processor


if __name__ == "__main__":
    dataloader, processor = load_training_data()
    model = prepare_model()
    train_model(model, dataloader, processor)
