import os
import re
import json
import torch
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import clip
from PIL import Image
from collections import Counter
import sys

# === Paths ===
IMAGE_ROOT = "E:/CDDMBench/dataset/images"  # ✅ Update to your dataset folder
CONV_JSON = "Crop_Disease_train_llava.json"
QNA_JSON = "Crop_Disease_train_qwenvl.json"
DIAGNOSIS_JSON = "disease_diagnosis.json"
KNOWLEDGE_JSON = "disease_knowledge.json"

# === Load data ===
with open(CONV_JSON, 'r', encoding='utf-8') as f:
    llava_data = json.load(f)
with open(QNA_JSON, 'r', encoding='utf-8') as f:
    qwenvl_data = json.load(f)
with open(DIAGNOSIS_JSON, 'r', encoding='utf-8') as f:
    diagnosis_data = json.load(f)
with open(KNOWLEDGE_JSON, 'r', encoding='utf-8') as f:
    knowledge_data = json.load(f)

# === Initialize models ===
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet50(weights="IMAGENET1K_V1")
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

clip_model, _ = clip.load("ViT-B/32", device=device)

# === Helper functions ===
def extract_image_path(conversations):
    """Extract image path from <img>...</img> in conversations"""
    for conv in conversations:
        if 'value' in conv and '<img>' in conv['value']:
            match = re.search(r'<img>(.*?)</img>', conv['value'])
            if match:
                return match.group(1)
    return None

def normalize_image_path(image_path):
    """Clean image path and get category"""
    image_path = image_path.lstrip("/")
    if image_path.startswith("dataset/images/"):
        image_path = image_path.replace("dataset/images/", "", 1)
    category = image_path.split("/")[0]
    return image_path, category

def extract_image_feature(image_path):
    image = Image.open(image_path).convert("RGB")
    image = img_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(image).squeeze().view(-1)
    return feat.cpu()

def extract_text_feature(text):
    # Truncate text safely for CLIP (max 77 tokens)
    text = text.strip()
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
    return text_features.squeeze(0).cpu()

def shorten_text(question="", answer="", limit=200):
    """Combine question + first 200 chars of answer"""
    return f"{question} {answer[:limit]}"

# === Build dataset ===
X, labels = [], []
missing = 0

print("🔄 Processing entries from all JSON files...")
debug_samples = []

# === Process LLaVA data ===
for item in llava_data:
    image_path = item.get('image')
    if image_path:
        image_path, category = normalize_image_path(image_path)
        debug_samples.append((image_path, category))

        conv_text = " ".join([c.get('value', '') for c in item.get('conversations', []) if isinstance(c, dict)])
        combined_text = shorten_text(conv_text, "")

        img_full_path = os.path.join(IMAGE_ROOT, image_path)
        if not os.path.exists(img_full_path):
            missing += 1
            continue

        img_feat = extract_image_feature(img_full_path)
        text_feat = extract_text_feature(combined_text)
        X.append(torch.cat([img_feat, text_feat]))
        labels.append(category)

# === Process QwenVL data ===
for item in qwenvl_data:
    image_path = extract_image_path(item.get('conversations', []))
    if image_path:
        image_path, category = normalize_image_path(image_path)
        debug_samples.append((image_path, category))

        conv_text = " ".join([c.get('value', '') for c in item.get('conversations', []) if isinstance(c, dict)])
        combined_text = shorten_text(conv_text, "")

        img_full_path = os.path.join(IMAGE_ROOT, image_path)
        if not os.path.exists(img_full_path):
            missing += 1
            continue

        img_feat = extract_image_feature(img_full_path)
        text_feat = extract_text_feature(combined_text)
        X.append(torch.cat([img_feat, text_feat]))
        labels.append(category)

# === Process diagnosis data ===
for item in diagnosis_data:
    image_path = item.get('image')
    if image_path:
        image_path, category = normalize_image_path(image_path)
        debug_samples.append((image_path, category))

        combined_text = shorten_text(item.get('question', ''), item.get('answer', ''))

        img_full_path = os.path.join(IMAGE_ROOT, image_path)
        if not os.path.exists(img_full_path):
            missing += 1
            continue

        img_feat = extract_image_feature(img_full_path)
        text_feat = extract_text_feature(combined_text)
        X.append(torch.cat([img_feat, text_feat]))
        labels.append(category)

# === Process knowledge data ===
for item in knowledge_data:
    image_path = item.get('image')
    if image_path:
        image_path, category = normalize_image_path(image_path)
        debug_samples.append((image_path, category))

        combined_text = shorten_text(item.get('question', ''), item.get('answer', ''))

        img_full_path = os.path.join(IMAGE_ROOT, image_path)
        if not os.path.exists(img_full_path):
            missing += 1
            continue

        img_feat = extract_image_feature(img_full_path)
        text_feat = extract_text_feature(combined_text)
        X.append(torch.cat([img_feat, text_feat]))
        labels.append(category)

# Debug info
print("\n🔍 First 5 samples:")
for img_path, cat in debug_samples[:5]:
    print(f"Path: {img_path} | Category: {cat}")

print(f"\n✅ Total processed: {len(X)}")
print(f"❌ Missing images: {missing}")

# === Check class distribution ===
print("\n📊 Checking class distribution...")
class_counts = Counter(labels)
print("Class counts:", class_counts)

if len(class_counts) <= 1:
    print("❌ Only one unique class found. Cannot train a classifier.")
    sys.exit(1)

if len(X) < 200:
    print("⚠️ WARNING: Very small dataset. Training results may be meaningless.")

# === Encode labels ===
print("🔤 Encoding labels...")
le = LabelEncoder()
y = torch.tensor(le.fit_transform(labels), dtype=torch.long)
x = torch.stack(X)

# === Create edges with KNN ===
print("🔗 Creating KNN graph edges...")
k = 10
x_np = x.numpy()
nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(x_np)
_, indices = nbrs.kneighbors(x_np)

edge_list = []
for i, neighbors in enumerate(indices):
    for j in neighbors[1:]:
        edge_list.append([i, j])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

# === Save graph ===
print("\n💾 Saving graph to 'cddm_graph.pt'...")
data = Data(x=x, edge_index=edge_index, y=y)
torch.save(data, "cddm_graph.pt")
print("✅ Graph saved successfully.")
