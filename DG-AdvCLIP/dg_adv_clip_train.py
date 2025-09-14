import clip
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

import os

from prompts_germany import class_name_promt_map as domain1_class_name_promt_map
from prompts_china import class_name_promt_map as domain2_class_name_promt_map


class DomainAdaptionDataset(Dataset):
    def __init__(self, germany_image_dir, china_image_dir, class_name_to_idx, german_prompts, chinese_prompts):
        self.germany_image_dir = germany_image_dir
        self.china_image_dir = china_image_dir
        self.germany_image_files = [f for f in os.listdir(germany_image_dir) if f.endswith(".png")]
        self.china_image_files = [f for f in os.listdir(china_image_dir) if f.endswith(".png")]
        self.class_name_to_idx = class_name_to_idx
        self.german_prompts = german_prompts
        self.chinese_prompts = chinese_prompts

    def __len__(self):
        return min(len(self.germany_image_files), len(self.china_image_files))
    
    def __getitem__(self, idx):
        # Load German image
        german_image_name = self.germany_image_files[idx % len(self.germany_image_files)]
        german_image_path = os.path.join(self.germany_image_dir, german_image_name)
        german_image = Image.open(german_image_path).convert("RGBA")
        class_name = " ".join(german_image_name.split("_")[:-1]) 
        class_name_idx = self.class_name_to_idx[class_name]
        german_text_prompt = self.german_prompts[class_name_idx]

        # Load Chinese image
        chinese_image_name = self.china_image_files[idx % len(self.china_image_files)]
        chinese_image_path = os.path.join(self.china_image_dir, chinese_image_name)
        chinese_image = Image.open(chinese_image_path).convert("RGBA")
        chinese_text_prompt = self.chinese_prompts[class_name_idx]

        return (
            preprocess(german_image), class_name_idx, german_text_prompt,
            preprocess(chinese_image), class_name_idx, chinese_text_prompt
        )
    

def compute_domain_descriminator_loss(domain1_logits, domain2_logits):
    B, K = domain2_logits.shape

    domain1_expanded = torch.cat([domain1_logits, torch.zeros_like(domain1_logits)], dim=1)  # [B, 2K]
    domain2_expanded = torch.cat([torch.zeros_like(domain2_logits), domain2_logits], dim=1)  # [B, 2K]

    domain_logits = torch.cat([domain1_expanded, domain2_expanded], dim=0)  # [2B, 2K]
    domain_probs = F.softmax(domain_logits, dim=1) 

    domain1_confidence = domain_probs[:B, :K].sum(dim=1)
    domain2_confidence = domain_probs[B:, K:].sum(dim=1)

    domain1_loss = -torch.log(domain1_confidence + 1e-6).mean()
    domain2_loss = -torch.log(domain2_confidence + 1e-6).mean()

    return (domain1_loss + domain2_loss)


def compute_category_confusion_loss(domain1_logits, domain2_logits, class_idx):
    B, K = domain2_logits.shape

    domain1_logits_det = domain1_logits.detach()
    domain2_logits_det = domain2_logits.detach()

    # Expand to 2K space
    domain1_expanded = torch.cat([domain1_logits_det, torch.zeros_like(domain1_logits_det)], dim=1)
    domain2_expanded = torch.cat([torch.zeros_like(domain2_logits_det), domain2_logits_det], dim=1)

    domain_logits = torch.cat([domain1_expanded, domain2_expanded], dim=0)  # [2B, 2K]
    domain_probs = F.softmax(domain_logits, dim=1)

    domain1_confuse = domain_probs[:B, class_idx + K]  
    domain2_confuse = domain_probs[B:, class_idx]      

    loss1 = -torch.log(domain1_confuse + 1e-6).mean()
    loss2 = -torch.log(domain2_confuse + 1e-6).mean()

    return (loss1 + loss2) / 2


def compute_domain_level_confusion_loss(domain1_logits, domain2_logits):
    B, K = domain2_logits.shape

    domain1_logits_det = domain1_logits.detach()
    domain2_logits_det = domain2_logits.detach()

    domain1_expanded = torch.cat([domain1_logits_det, torch.zeros_like(domain1_logits_det)], dim=1)  # [B, 2K]
    domain2_expanded = torch.cat([torch.zeros_like(domain2_logits_det), domain2_logits_det], dim=1)  # [B, 2K]

    domain_logits = torch.cat([domain1_expanded, domain2_expanded], dim=0)  # [2B, 2K]
    domain_probs = F.softmax(domain_logits, dim=1)

    domain1_confuse = domain_probs[:B, K:].sum(dim=1)  
    domain2_confuse = domain_probs[B:, :K].sum(dim=1)  

    loss1 = -torch.log(domain1_confuse + 1e-6).mean()
    loss2 = -torch.log(domain2_confuse + 1e-6).mean()

    return (loss1 + loss2) / 2

# unused
def compute_conditional_entropy_loss(domain1_logits, domain2_logits):
    """
    domain1_logits: logits for German features, shape [B, K]
    domain2_logits: logits for Chinese features, shape [B, K]
    Returns: scalar loss value (minimize this)
    """

    B, K = domain1_logits.shape

    # Expand into 2K logits as per your domain classifier structure
    domain1_full = torch.cat([domain1_logits, torch.zeros_like(domain1_logits)], dim=1)  # [B, 2K]
    domain2_full = torch.cat([torch.zeros_like(domain2_logits), domain2_logits], dim=1)  # [B, 2K]

    probs1 = F.softmax(domain1_full, dim=1)
    probs2 = F.softmax(domain2_full, dim=1)

    # Get only the respective domain-specific parts
    domain1_probs = probs1[:, :K]        # German domain predictions
    domain2_probs = probs2[:, K:]        # Chinese domain predictions

    entropy1 = - (domain1_probs * torch.log(domain1_probs + 1e-6)).sum(dim=1).mean()
    entropy2 = - (domain2_probs * torch.log(domain2_probs + 1e-6)).sum(dim=1).mean()

    return 0.5 * (entropy1 + entropy2)


device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
# load resnet model
model, preprocess = clip.load("RN50", device=device)

# For Germany dataset
path_to_train_data_domain1 = "local/latest/Germany/train_germany_mix"
domain1_image_files = [f for f in os.listdir(path_to_train_data_domain1) if f.endswith(".png")]
class_names = sorted(domain1_class_name_promt_map.keys())
domain1_text_prompts = [". ".join(domain1_class_name_promt_map[name]) for name in class_names]
num_classes = len(class_names)
class_name_to_idx = {name: idx for idx, name in enumerate(class_names)}
domain1_idx_to_text_promts = {idx: ". ".join(domain1_class_name_promt_map[name]) for name, idx in class_name_to_idx.items()}

# For China dataset
path_to_train_data_domain2 = "local/latest/China/cn_train_mix"
domain2_image_files = [f for f in os.listdir(path_to_train_data_domain2) if f.endswith(".png")]
domain2_class_names = sorted(domain2_class_name_promt_map.keys())
domain2_text_prompts = [". ".join(domain2_class_name_promt_map[name]) for name in domain2_class_names]
domain2_class_name_to_idx = {name: idx for idx, name in enumerate(domain2_class_names)}
domain2_idx_to_text_promts = {idx: ". ".join(domain2_class_name_promt_map[name]) for name, idx in domain2_class_name_to_idx.items()}

train_dataset = DomainAdaptionDataset(path_to_train_data_domain1, path_to_train_data_domain2, 
                                      class_name_to_idx, 
                                      domain1_text_prompts, domain2_text_prompts)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for param in model.visual.parameters():
    param.requires_grad = False

# unfreeze attention pooling layer
for param in model.visual.attnpool.parameters():
    param.requires_grad = True

# FC classifier
domain1_classifier = nn.Linear(model.visual.output_dim, num_classes).to(device)
domain2_classifier = nn.Linear(model.visual.output_dim, num_classes).to(device)

# Initialize the classifier with text prompts
class_name_inputs = clip.tokenize(class_names).to(device)
with torch.no_grad():
    class_name_features = model.encode_text(class_name_inputs)
    class_name_features /= class_name_features.norm(dim=-1, keepdim=True)
    domain1_classifier.weight.data = class_name_features
    domain1_classifier.bias.data.zero_()
    domain2_classifier.weight.data = class_name_features
    domain2_classifier.bias.data.zero_()


optimizer_domain_desc = torch.optim.Adam(
    list(domain1_classifier.parameters()) + list(domain2_classifier.parameters()),
    lr=1e-4
)
optimizer_attnpool = torch.optim.Adam(
    model.visual.attnpool.parameters(),
    lr=1e-5 
)

num_epochs = 20

scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_domain_desc, T_max=num_epochs)
scheduler_attnpool = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_attnpool, T_max=num_epochs)

domain1_loss_criterion = nn.CrossEntropyLoss()
domain2_loss_criterion = nn.CrossEntropyLoss()
classifier_loss_criterion = nn.CrossEntropyLoss()

all_domain1_losses = []
all_domain2_losses = []
all_domain_descriminator_loss = []
all_category_confusion_loss = []
all_domain_level_confusion_loss = []
all_conditional_entropy_loss = []
all_classification_losses = []
all_domain1_losses = []
all_domain2_losses = []

with torch.no_grad():
    domain1_text_inputs = clip.tokenize(domain1_text_prompts, truncate=True).to(device)
    domain1_text_features = model.encode_text(domain1_text_inputs)
    domain1_text_features = domain1_text_features / domain1_text_features.norm(dim=-1, keepdim=True)
    domain1_text_features = domain1_text_features.detach()
    domain2_text_inputs = clip.tokenize(domain2_text_prompts, truncate=True).to(device) 
    domain2_text_features = model.encode_text(domain2_text_inputs)
    domain2_text_features = domain2_text_features / domain2_text_features.norm(dim=-1, keepdim=True)
    domain2_text_features = domain2_text_features.detach()


for epoch in range(num_epochs):
    model.eval()
    model.visual.attnpool.train()
    domain1_classifier.train()
    domain2_classifier.train()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_domain_descriminator_training_loss = 0.0
    total_category_confusion_loss = 0.0
    total_domain_level_confusion_loss = 0.0
    total_domain1_loss = 0.0
    total_domain2_loss = 0.0

    for (
    domain1_images, class_idx_domain1, domain1_texts,
    domain2_images, class_idx_domain2, domain2_texts
    ) in tqdm(train_loader):

  
        K = num_classes
        B = class_idx_domain1.size(0)
        domain1_images = domain1_images.to(device)
        domain2_images = domain2_images.to(device)
        class_idx = class_idx_domain1.to(device) 

        domain1_image_features = model.encode_image(domain1_images)
        domain1_image_features = domain1_image_features / domain1_image_features.norm(dim=-1, keepdim=True)

        domain2_image_features = model.encode_image(domain2_images)
        domain2_image_features = domain2_image_features / domain2_image_features.norm(dim=-1, keepdim=True)

        similarity_domain1 = (100.0 * domain1_image_features @ domain1_text_features.T).softmax(dim=-1)
        similarity_domain2 = (100.0 * domain2_image_features @ domain2_text_features.T).softmax(dim=-1)
        
        one_hot_encoding_domain1 = torch.zeros(B*2, K, device=device)
        one_hot_encoding_domain1.scatter_(1, class_idx_domain1.unsqueeze(1), 1.0)

        one_hot_encoding_domain2 = torch.zeros(B*2, K, device=device)
        one_hot_encoding_domain2.scatter_(1, class_idx_domain2.unsqueeze(1), 1.0)

        classification_loss = (
            classifier_loss_criterion(similarity_domain1, class_idx) + 
            classifier_loss_criterion(similarity_domain2, class_idx)
        ) / 2

        domain1_logits = domain1_classifier(torch.cat([domain1_text_features[class_idx_domain1], domain1_image_features], dim=0))
        domain2_logits = domain2_classifier(torch.cat([domain2_text_features[class_idx_domain2], domain2_image_features], dim=0))

        category_confusion_loss = compute_category_confusion_loss(domain1_logits, domain2_logits, class_idx)
        domain_level_confusion_loss = compute_domain_level_confusion_loss(domain1_logits, domain2_logits)

        attention_pool_loss = category_confusion_loss + domain_level_confusion_loss + classification_loss

        optimizer_attnpool.zero_grad()
        attention_pool_loss.backward()
        optimizer_attnpool.step()
        scheduler_attnpool.step()

        domain1_logits_dom_disc = domain1_classifier(torch.cat([domain1_text_features[class_idx_domain1], domain1_image_features.detach()], dim=0))
        domain2_logits_dom_disc = domain2_classifier(torch.cat([domain2_text_features[class_idx_domain2], domain2_image_features.detach()], dim=0))

        domain1_loss = domain1_loss_criterion(domain1_logits_dom_disc, one_hot_encoding_domain1)
        domain2_loss = domain2_loss_criterion(domain2_logits_dom_disc, one_hot_encoding_domain2)

        domain_descriminator_loss = compute_domain_descriminator_loss(domain1_logits_dom_disc, domain2_logits_dom_disc)
        # conditional_entropy_loss = compute_conditional_entropy_loss(domain1_logits, domain2_logits)

        domain_descriminator_training_loss = domain1_loss + domain2_loss + domain_descriminator_loss

        optimizer_domain_desc.zero_grad()
        domain_descriminator_training_loss.backward(retain_graph=True)
        optimizer_domain_desc.step()
        scheduler_cls.step()

        total_loss += attention_pool_loss.item() + domain_descriminator_loss.item()
        total_classification_loss += classification_loss.item()
        total_domain_descriminator_training_loss += domain_descriminator_training_loss.item()
        total_category_confusion_loss += category_confusion_loss.item()
        total_domain_level_confusion_loss += domain_level_confusion_loss.item()
        total_domain1_loss += domain1_loss.item()
        total_domain2_loss += domain2_loss.item()


    avg_loss = total_loss / len(train_loader)
    avg_classification_loss = total_classification_loss / len(train_loader)
    avg_domain_descriminator_training_loss = total_domain_descriminator_training_loss / len(train_loader)
    avg_category_confusion_loss = total_category_confusion_loss / len(train_loader)
    avg_domain_level_confusion_loss = total_domain_level_confusion_loss / len(train_loader)
    all_classification_losses.append(avg_classification_loss)
    all_domain_descriminator_loss.append(avg_domain_descriminator_training_loss)
    all_category_confusion_loss.append(avg_category_confusion_loss)
    all_domain_level_confusion_loss.append(avg_domain_level_confusion_loss)
    all_domain1_losses.append(total_domain1_loss / len(train_loader))
    all_domain2_losses.append(total_domain2_loss / len(train_loader))
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


epochs = list(range(1, len(all_classification_losses) + 1))

plt.figure(figsize=(12, 6))
plt.plot(epochs, all_classification_losses, label='Classifier Loss', color='blue')
plt.plot(epochs, all_domain_descriminator_loss, label='Domain Discriminator Loss', color='orange')
plt.plot(epochs, all_category_confusion_loss, label='Category Confusion Loss', color='green')
plt.plot(epochs, all_domain_level_confusion_loss, label='Domain Level Confusion Loss', color='red')
plt.plot(epochs, all_domain1_losses, label='Domain 1 Loss', color='cyan')
plt.plot(epochs, all_domain2_losses, label='Domain 2 Loss', color='magenta')
# plt.plot(epochs, all_conditional_entropy_loss, label='Conditional Entropy Loss', color='purple')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Losses over Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save attention pool weights and classifier
save_path = "fine_tuned_clip.pth"

torch.save({
    'attnpool_state_dict': model.visual.attnpool.state_dict(),
    'image_classifier_state_dict': domain1_classifier.state_dict(),
    'text_classifier_state_dict': domain2_classifier.state_dict(),
}, save_path)

print(f"✅ Model saved to {save_path}")

