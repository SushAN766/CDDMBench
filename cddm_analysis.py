# cddm_analysis.py
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import umap
from torch_geometric.nn import GCNConv
import os

# Paths
GRAPH_PATH = "cddm_graph.pt"
MODEL_PATH = "./output/gnn_model_best.pth"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- Load Graph -------- #
print(f"Loading graph from {GRAPH_PATH}...")
graph = torch.load(GRAPH_PATH, weights_only=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
graph = graph.to(device)

labels = graph.y.cpu().numpy()

# -------- Define GCN Model (same as training) -------- #
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_embeddings=False):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        if return_embeddings:
            return x
        x = self.conv2(x, edge_index)
        return x

# Load model
model = GCN(graph.num_node_features, 128, graph.y.max().item() + 1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------- Get Predictions & Embeddings -------- #
with torch.no_grad():
    out = model(graph.x, graph.edge_index)
    pred = out.argmax(dim=1).cpu().numpy()
    embeddings = model(graph.x, graph.edge_index, return_embeddings=True).cpu().numpy()

# -------- 1. Graph Structure Visualization -------- #
print("Generating graph structure visualization...")
G = to_networkx(graph)
plt.figure(figsize=(8, 8))
nx.draw(G, node_size=15, node_color=labels, cmap='tab20', edge_color='gray')
plt.title("Graph Structure")
plt.savefig(f"{OUTPUT_DIR}/graph_structure.png")
plt.close()

# -------- 2. Class Distribution -------- #
print("Generating class distribution chart...")
unique_labels, counts = torch.unique(graph.y.cpu(), return_counts=True)
plt.figure(figsize=(6, 4))
plt.bar(unique_labels.numpy(), counts.numpy())
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png")
plt.close()

# -------- 3. Confusion Matrix -------- #
print("Generating confusion matrix...")
cm = confusion_matrix(labels, pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='viridis')
plt.title("Confusion Matrix")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()

# -------- 4. UMAP Visualization -------- #
print("Generating UMAP embedding visualization...")
umap_result = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(embeddings)
plt.figure(figsize=(6, 6))
plt.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='tab20', s=10)
plt.title("UMAP Embedding")
plt.savefig(f"{OUTPUT_DIR}/umap_embedding.png")
plt.close()

print(f"\n✅ All analytics saved in {OUTPUT_DIR}")
