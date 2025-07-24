#Features of New cddm_graph_builder1.py (Advanced)
#✔ Graph construction options:

#KNN Graph (default)

#Cosine Similarity Threshold Graph

#Heterogeneous Graph with category nodes
#✔ Combines multimodal features (Image + Text)
#✔ Includes debug info for edge statistics
#✔ Saves graph in PyTorch Geometric Data format




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
import numpy as np
import sys

# === CONFIG ===
IMAGE_ROOT = "E:/CDDMBench/dataset/images"
CONV_JSON = "Crop_Disease_train_llava.json"
QNA_JSON = "Crop_Disease_train_qwenvl.json"
DIAGNOSIS_JSON = "disease_diagnosis.json"
KNOWLEDGE_JSON = "disease_knowledge.json"
GRAPH_SAVE_PATH = "cddm_graph_data.pt"

GRAPH_MODE = "heterogeneous"  # options: "knn", "threshold", "heterogeneous"
K_NEIGHBORS = 10
SIM_THRESHOLD = 0.8

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load data ===
with open(CONV_JSON, 'r', encoding='utf-8') as f:
    llava_data = json.load(f)
with open(QNA_JSON, 'r', encoding='utf-8') as f:
    qwenvl_data = json.load(f)
with open(DIAGNOSIS_JSON, 'r', encoding='utf-8') as f:
    diagnosis_data = json.load(f)
with open(KNOWLEDGE_JSON, 'r', encoding='utf-8') as f:
    knowledge_data = json.load(f)

# === Models ===
resnet = models.resnet50(weights="IMAGENET1K_V1")
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

img_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

clip_model, _ = clip.load("ViT-B/32", device=device)

# === Helper functions ===
def extract_image_path(conversations):
    for conv in conversations:
        if 'value' in conv and '<img>' in conv['value']:
            match = re.search(r'<img>(.*?)</img>', conv['value'])
            if match:
                return match.group(1)
    return None

def normalize_image_path(image_path):
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
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
    return text_features.squeeze(0).cpu()

def shorten_text(q="", a="", limit=200):
    return f"{q} {a[:limit]}"

# === Build dataset ===
X, labels = [], []
missing = 0
debug_samples = []

print("🔄 Processing all JSON files...")

# Process all data files
def process_entry(image_path, category, text):
    global missing
    img_full_path = os.path.join(IMAGE_ROOT, image_path)
    if not os.path.exists(img_full_path):
        missing += 1
        return
    img_feat = extract_image_feature(img_full_path)
    text_feat = extract_text_feature(text)
    X.append(torch.cat([img_feat, text_feat]))
    labels.append(category)

# LLaVA
for item in llava_data:
    image_path = item.get('image')
    if image_path:
        img_path, category = normalize_image_path(image_path)
        conv_text = " ".join([c.get('value', '') for c in item.get('conversations', [])])
        process_entry(img_path, category, shorten_text(conv_text, ""))

# QwenVL
for item in qwenvl_data:
    image_path = extract_image_path(item.get('conversations', []))
    if image_path:
        img_path, category = normalize_image_path(image_path)
        conv_text = " ".join([c.get('value', '') for c in item.get('conversations', [])])
        process_entry(img_path, category, shorten_text(conv_text, ""))

# Diagnosis
for item in diagnosis_data:
    image_path = item.get('image')
    if image_path:
        img_path, category = normalize_image_path(image_path)
        process_entry(img_path, category, shorten_text(item.get('question', ''), item.get('answer', '')))

# Knowledge
for item in knowledge_data:
    image_path = item.get('image')
    if image_path:
        img_path, category = normalize_image_path(image_path)
        process_entry(img_path, category, shorten_text(item.get('question', ''), item.get('answer', '')))

# Debug info
print(f"\n✅ Total processed: {len(X)}")
print(f"❌ Missing images: {missing}")

class_counts = Counter(labels)
print("\n📊 Class distribution:", class_counts)

if len(class_counts) <= 1:
    print("❌ Only one unique class found. Cannot train a classifier.")
    sys.exit(1)

# Encode labels
le = LabelEncoder()
y = torch.tensor(le.fit_transform(labels), dtype=torch.long)
x = torch.stack(X)
x_np = x.numpy()

# === Graph construction ===
print(f"\n🔗 Building {GRAPH_MODE.upper()} graph...")
edge_index_list = []

if GRAPH_MODE == "knn":
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS+1, metric='cosine').fit(x_np)
    _, indices = nbrs.kneighbors(x_np)
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            edge_index_list.append([i, j])

elif GRAPH_MODE == "threshold":
    sim_matrix = np.dot(x_np, x_np.T) / (np.linalg.norm(x_np, axis=1)[:, None] * np.linalg.norm(x_np, axis=1))
    edges = np.argwhere(sim_matrix > SIM_THRESHOLD)
    for i, j in edges:
        if i != j:
            edge_index_list.append([i, j])

elif GRAPH_MODE == "heterogeneous":
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS+1, metric='cosine').fit(x_np)
    _, indices = nbrs.kneighbors(x_np)
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            edge_index_list.append([i, j])
    num_samples = len(X)
    num_classes = len(le.classes_)
    for i, label in enumerate(y):
        class_node = num_samples + int(label)
        edge_index_list.append([i, class_node])
        edge_index_list.append([class_node, i])
    print(f"✅ Added {num_classes} category nodes.")

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

# === Save graph ===
data = Data(x=x, edge_index=edge_index, y=y)
torch.save(data, GRAPH_SAVE_PATH)
print(f"\n✅ Graph saved to {GRAPH_SAVE_PATH}")
print(f"ℹ️ Nodes: {data.num_nodes}, Edges: {data.num_edges}")
