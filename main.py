# app.py - Multimodal Crop Disease Predictor (Final Working Version)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import torch.nn as nn
import clip # type: ignore
import json
import numpy as np
import os
import re

# --- Helper to extract category from path (Used for JSON lookup keys) ---
def normalize_image_path_and_extract_category(image_path):
    """Cleans up file path prefixes and extracts the 'Crop,Disease' category."""
    image_path = image_path.lstrip("/")
    
    # Clean common prefixes used in your training data paths
    if image_path.startswith("dataset/images/"):
        image_path = image_path.replace("dataset/images/", "", 1)
    if image_path.startswith("home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv/"):
        image_path = image_path.replace("home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv/", "", 1)

    # The category is the part before the final file name (e.g., 'Rice,Blast')
    category = image_path.split("/")[0] 
    return category


# --- 0. Load Configuration and Knowledge Data ---
try:
    with open("cddm_config.json", 'r', encoding='utf-8') as f:
        CONFIG = json.load(f)
    
    # Load raw JSON files
    with open("disease_knowledge.json", 'r', encoding='utf-8') as f:
        KNOWLEDGE_DATA_RAW = json.load(f)
        
    with open("disease_diagnosis.json", 'r', encoding='utf-8') as f:
        DIAGNOSIS_DATA_RAW = json.load(f)

    # --- CRITICAL FIX: Create Lookups using the FIRST answer found for each category ---
    # We only store the first answer found for a category to prevent repetition.
    KNOWLEDGE_LOOKUP = {}
    for item in KNOWLEDGE_DATA_RAW:
        image_path = item.get('image')
        answer = item.get('answer')
        if image_path and answer:
            category = normalize_image_path_and_extract_category(image_path)
            if category not in KNOWLEDGE_LOOKUP:
                KNOWLEDGE_LOOKUP[category] = answer

    DIAGNOSIS_LOOKUP = {}
    for item in DIAGNOSIS_DATA_RAW:
        image_path = item.get('image')
        answer = item.get('answer')
        if image_path and answer:
            category = normalize_image_path_and_extract_category(image_path)
            if category not in DIAGNOSIS_LOOKUP: 
                DIAGNOSIS_LOOKUP[category] = answer
    
    # Configuration constants
    NODE_FEATURE_SIZE = CONFIG['NODE_FEATURE_SIZE']
    NUM_CLASSES = CONFIG['NUM_CLASSES']
    CLASS_NAMES = CONFIG['CLASS_NAMES']
    MODEL_PATH = 'output/gnn_model_best.pth' if os.path.exists('output/gnn_model_best.pth') else 'gnn_model_best.pth'
    GCN_HIDDEN_CHANNELS = 128
    IMAGE_SIZE = 224
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

except FileNotFoundError as e:
    print(f"❌ CRITICAL: Configuration file not found. Ensure all JSON files are present.")
    exit(1)


# ==============================================================================
# 1 & 2. Model Definitions, Feature Extraction, and Graph Creation
# ==============================================================================

# --- A. Multimodal Feature Extractor Setup ---
resnet = models.resnet50(weights="IMAGENET1K_V1")
RESNET_EXTRACTOR = torch.nn.Sequential(*list(resnet.children())[:-1])
RESNET_EXTRACTOR.eval().to(DEVICE)

CLIP_MODEL, _ = clip.load("ViT-B/32", device=DEVICE)

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# --- B. GNN Architecture (GCN) ---
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# --- C. GNN Model Loading ---
try:
    GNN_MODEL = GCN(
        in_channels=NODE_FEATURE_SIZE, 
        hidden_channels=GCN_HIDDEN_CHANNELS, 
        out_channels=NUM_CLASSES # Corrected to 59
    )
    GNN_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    GNN_MODEL.to(DEVICE)
    GNN_MODEL.eval()
    MODEL_LOADED = True
    print(f"✅ GNN Model loaded successfully. Ready for inference on {DEVICE}.")
except Exception as e:
    GNN_MODEL = None
    MODEL_LOADED = False
    print(f"❌ Error loading GNN model weights: {e}")

# --- Feature Extraction Helpers ---
def extract_image_feature(image: Image.Image):
    image_tensor = IMG_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = RESNET_EXTRACTOR(image_tensor).squeeze().view(-1) 
    return feat.cpu()

def extract_text_feature(text: str):
    text = text.strip()
    tokens = clip.tokenize([text], truncate=True).to(DEVICE)
    with torch.no_grad():
        text_features = CLIP_MODEL.encode_text(tokens)
    return text_features.squeeze(0).cpu()

def create_inference_graph(image: Image.Image, text_description: str):
    img_feat = extract_image_feature(image)
    text_feat = extract_text_feature(text_description)
    combined_feat = torch.cat([img_feat, text_feat]) # 2560 features
    x = combined_feat.unsqueeze(0).to(DEVICE)
    edge_index = torch.empty((2, 0), dtype=torch.long).to(DEVICE) 
    return Data(x=x, edge_index=edge_index)


# ==============================================================================
# 3. Prediction Function (Returns 3 Outputs)
# ==============================================================================

def predict_multimodal_disease(input_image: Image.Image, text_input: str):
    
    if not MODEL_LOADED:
        return {"ERROR: Model not loaded.": 0.0}, "Model initialization failed.", "Model initialization failed."
    
    graph_data = create_inference_graph(input_image, text_input)

    # GNN Inference
    with torch.no_grad():
        output = GNN_MODEL(graph_data.x, graph_data.edge_index)

    # Classification & Top Category Extraction
    probabilities = F.softmax(output[0], dim=0)
    # Changed to topk(1) to satisfy the user request for exact single prediction
    top_p, top_class = probabilities.topk(1, dim=0) 
    
    results_dict = {}
    top_prediction_category = "" 
    
    for i in range(top_p.size(0)):
        index = top_class[i].item()
        confidence = top_p[i].item()
        
        predicted_label = CLASS_NAMES[index] 
        
        if i == 0:
            top_prediction_category = predicted_label
            
        # Format for Gradio Label output
        if ',' in predicted_label:
            crop_name, disease_name = predicted_label.split(',', 1) 
        else:
            crop_name = "Crop"
            disease_name = predicted_label
            
        display_label = f"Crop: {crop_name.replace('_', ' ').strip()} | Disease: {disease_name.replace('_', ' ').strip()}"
        results_dict[display_label] = float(confidence)

    # --- LOOKUP 1: Management Knowledge ---
    management_answer = KNOWLEDGE_LOOKUP.get(
        top_prediction_category, 
        f"Management knowledge not found for: '{top_prediction_category}'. Please ensure this key exists in disease_knowledge.json."
    )
    
    # --- LOOKUP 2: Diagnosis/Symptoms ---
    diagnosis_answer = DIAGNOSIS_LOOKUP.get(
        top_prediction_category, 
        f"Diagnosis/Symptoms not found for: '{top_prediction_category}'. Please ensure this key exists in disease_diagnosis.json."
    )
    
    # Returns: [Classification Dict, Management Text, Diagnosis Text]
    return results_dict, management_answer, diagnosis_answer


# ==============================================================================
# 4. Gradio Interface Setup (CRITICAL: num_top_classes=1)
# ==============================================================================

text_context_input = gr.Textbox(
    label="Provide Text Context (Default or Custom Question)",
    value="What are the specific features and control measures for this disease?" 
)

management_output = gr.Textbox(
    label="🔍 Disease Management & Control Measures (Knowledge)", 
    interactive=False, 
    lines=8, 
    show_copy_button=True
)

diagnosis_output = gr.Textbox(
    label="🔬 Detailed Diagnosis / Symptoms (Diagnosis)", 
    interactive=False, 
    lines=4, 
    show_copy_button=True
)

iface = gr.Interface(
    fn=predict_multimodal_disease,
    inputs=[
        gr.Image(type="pil", label="Upload Crop Leaf Image", image_mode="RGB"),
        text_context_input 
    ],
    outputs=[
        # Displays only the top 1 prediction and its confidence
        gr.Label(label="Classification Result (Confidence Score)", num_top_classes=1), 
        management_output,
        diagnosis_output
    ],
    title="🌿 Multimodal Crop Disease Predictor (GNN Classifier)",
    description=(
        f"This model uses **ResNet50 + CLIP** features, classified by a **GCN**. "
        f"Total Classes: {NUM_CLASSES}."
    )
)

if __name__ == '__main__':
    iface.launch(inbrowser=True)