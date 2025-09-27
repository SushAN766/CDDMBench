# app.py - Crop Disease Multimodal Predictor

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

# --- 0. Load Configuration ---
try:
    with open("cddm_config.json", 'r', encoding='utf-8') as f:
        CONFIG = json.load(f)
    
    NODE_FEATURE_SIZE = CONFIG['NODE_FEATURE_SIZE']
    NUM_CLASSES = CONFIG['NUM_CLASSES']
    CLASS_NAMES = CONFIG['CLASS_NAMES']
    
    # Extract the LabelEncoder logic to correctly parse the crop and disease
    # The class names are in the format "CropName,DiseaseName"
    
except FileNotFoundError:
    print("❌ CRITICAL: 'cddm_config.json' not found. Run 'setup_config.py' first.")
    exit(1)


# ==============================================================================
# 1. Model Definitions & Initialization
# ==============================================================================

MODEL_PATH = 'gnn_model_best.pth'
GCN_HIDDEN_CHANNELS = 128
IMAGE_SIZE = 224 # Used for ResNet input
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- A. Multimodal Feature Extractor Setup ---
# ResNet50 for Image Features (2048 features)
resnet = models.resnet50(weights="IMAGENET1K_V1")
RESNET_EXTRACTOR = torch.nn.Sequential(*list(resnet.children())[:-1])
RESNET_EXTRACTOR.eval().to(DEVICE)

# CLIP for Text Features (512 features)
CLIP_MODEL, _ = clip.load("ViT-B/32", device=DEVICE)

# Image Transformations for ResNet
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# --- B. GNN Architecture (Copied EXACTLY from your training code) ---
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
        out_channels=NUM_CLASSES
    )
    GNN_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    GNN_MODEL.to(DEVICE)
    GNN_MODEL.eval()
    MODEL_LOADED = True
    print(f"✅ GNN Model '{MODEL_PATH}' loaded successfully on {DEVICE}.")
except Exception as e:
    GNN_MODEL = None
    MODEL_LOADED = False
    print(f"❌ Error loading GNN model: {e}")


# ==============================================================================
# 2. Multimodal Feature Extraction and Graph Creation
# ==============================================================================

def extract_image_feature(image: Image.Image):
    """Extracts 2048-dim ResNet feature from the image."""
    image_tensor = IMG_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        # ResNet output shape: [1, 2048, 1, 1] -> [2048]
        feat = RESNET_EXTRACTOR(image_tensor).squeeze().view(-1) 
    return feat.cpu()

def extract_text_feature(text: str):
    """Extracts 512-dim CLIP feature from the text."""
    text = text.strip()
    tokens = clip.tokenize([text], truncate=True).to(DEVICE)
    with torch.no_grad():
        # CLIP output shape: [1, 512] -> [512]
        text_features = CLIP_MODEL.encode_text(tokens)
    return text_features.squeeze(0).cpu()

def create_inference_graph(image: Image.Image, text_description: str):
    """Creates a single-node graph (the entire image/text pair) for inference."""
    
    # 1. Feature Extraction
    img_feat = extract_image_feature(image)   # 2048 features
    text_feat = extract_text_feature(text_description) # 512 features
    
    # Combine features: shape [2560]
    combined_feat = torch.cat([img_feat, text_feat]) 
    
    # The GNN model was trained on a graph where each image/text pair 
    # is represented as a *node* in a large graph. For prediction on a single 
    # unseen sample, we create a temporary "micro-graph" with one node.
    
    # Node Features (x): [1 node, 2560 features]
    x = combined_feat.unsqueeze(0).to(DEVICE)
    
    # Edge Index (edge_index): Since the prediction is for a single, unseen node 
    # that is not connected to the training graph, we use empty edges. 
    # The GNN will perform a self-loop message passing if configured, 
    # but primarily it will use the feature vector.
    edge_index = torch.empty((2, 0), dtype=torch.long).to(DEVICE) 

    # Create the PyTorch Geometric Data object
    graph_data = Data(x=x, edge_index=edge_index)
    
    return graph_data


# ==============================================================================
# 3. Prediction Function
# ==============================================================================

def predict_multimodal_disease(input_image: Image.Image, text_input: str):
    """
    Predicts the crop disease using the image and an associated text description.
    """
    if not MODEL_LOADED:
        return {"ERROR: Model not loaded. Check console for startup details.": 0.0} 
    
    # 1. Create the Graph Node
    graph_data = create_inference_graph(input_image, text_input)

    # 2. GNN Inference
    with torch.no_grad():
        # The GNN outputs [num_nodes, num_classes] -> [1, 60]
        output = GNN_MODEL(graph_data.x, graph_data.edge_index)

    # 3. Get the final prediction
    probabilities = F.softmax(output[0], dim=0)
    top_p, top_class = probabilities.topk(3, dim=0)
    
    # 4. Format results
    results_dict = {}
    for i in range(top_p.size(0)):
        index = top_class[i].item()
        confidence = top_p[i].item()
        
        predicted_label = CLASS_NAMES[index]
        
        # Split by comma (e.g., 'Apple,Alternaria Blotch')
        if ',' in predicted_label:
            crop_name, disease_name = predicted_label.split(',', 1) 
        else:
            crop_name = "Crop"
            disease_name = predicted_label
            
        display_label = f"Crop: {crop_name.replace('_', ' ').strip()} | Disease: {disease_name.replace('_', ' ').strip()}"
        results_dict[display_label] = float(confidence)

    return results_dict


# ==============================================================================
# 4. Gradio Interface Setup
# ==============================================================================

iface = gr.Interface(
    fn=predict_multimodal_disease,
    inputs=[
        gr.Image(type="pil", label="Upload Crop Leaf Image", image_mode="RGB"),
        gr.Textbox(
            label="Provide Text Context (e.g., Question/Diagnosis/Knowledge)",
            value="Describe the disease visible on the leaf."
        )
    ],
    outputs=gr.Label(
        label="Prediction Result (Confidence Score)",
        num_top_classes=3
    ),
    title="🌿 Multimodal Crop Disease Predictor (GNN Classifier)",
    description=(
        f"This model uses **ResNet50 + CLIP** features, classified by a **GCN** trained on your custom graph."
        f"It requires both an **image** and a **text description** to classify among {NUM_CLASSES} diseases."
    )
)

if __name__ == '__main__':
    iface.launch(inbrowser=True)