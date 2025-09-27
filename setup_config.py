# setup_config.py
import json
from sklearn.preprocessing import LabelEncoder
import os
import re

# === Paths (Adjust these if needed) ===
CONV_JSON = "Crop_Disease_train_llava.json"
QNA_JSON = "Crop_Disease_train_qwenvl.json"
DIAGNOSIS_JSON = "disease_diagnosis.json"
KNOWLEDGE_JSON = "disease_knowledge.json"

# === Helper functions from your training code ===
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

# === Load and Extract Labels ===
labels = []
datasets = [CONV_JSON, QNA_JSON, DIAGNOSIS_JSON, KNOWLEDGE_JSON]
image_key = {'Crop_Disease_train_llava.json': 'image', 'disease_diagnosis.json': 'image', 'disease_knowledge.json': 'image'}

for file_path in datasets:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    is_conv = 'qwenvl' in file_path or 'llava' in file_path
    
    for item in data:
        if is_conv:
            image_path = item.get('image') if 'llava' in file_path else extract_image_path(item.get('conversations', []))
        else:
            image_path = item.get('image')
        
        if image_path:
            _, category = normalize_image_path(image_path)
            labels.append(category)

# === Get Unique Classes and Save ===
le = LabelEncoder()
le.fit(labels)
sorted_class_names = list(le.classes_)

config = {
    "NODE_FEATURE_SIZE": 2560 + 512, # ResNet50 (2048) + CLIP (512)
    "NUM_CLASSES": len(sorted_class_names),
    "CLASS_NAMES": sorted_class_names
}

with open("cddm_config.json", 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4)
    
print(f"✅ Configuration saved to cddm_config.json with {len(sorted_class_names)} classes.")
print(f"   Feature size is {config['NODE_FEATURE_SIZE']} (2048 from ResNet + 512 from CLIP).")

# RUN THIS SCRIPT ONCE: python setup_config.py