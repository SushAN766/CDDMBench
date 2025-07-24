# cddm_visualizer.py
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_geometric.nn import GCNConv
import os

# -------- LOAD GRAPH & MODEL -------- #
GRAPH_PATH = "cddm_graph.pt"
MODEL_PATH = "./output/gnn_model_best.pth"

print(f"Loading graph from {GRAPH_PATH}...")
graph = torch.load(GRAPH_PATH, weights_only=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
graph = graph.to(device)

# Ensure output folder exists
os.makedirs("./output", exist_ok=True)

# -------- DEFINE SAME GCN MODEL -------- #
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_embeddings=False):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        if return_embeddings:
            return x  # Return embeddings from hidden layer
        x = self.conv2(x, edge_index)
        return x

model = GCN(graph.num_node_features, 128, graph.y.max().item() + 1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------- GET NODE EMBEDDINGS -------- #
with torch.no_grad():
    embeddings = model(graph.x, graph.edge_index, return_embeddings=True).cpu()

labels = graph.y.cpu().numpy()

# -------- APPLY DIMENSIONALITY REDUCTION -------- #
print("Applying PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)

print("Applying t-SNE (this may take some time)...")
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, verbose=1)
tsne_result = tsne.fit_transform(embeddings)


# -------- PLOT VISUALIZATION -------- #
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# PCA plot
axes[0].scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab20', s=10)
axes[0].set_title('PCA Embedding')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# t-SNE plot
axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab20', s=10)
axes[1].set_title('t-SNE Embedding')
axes[1].set_xlabel('Dim 1')
axes[1].set_ylabel('Dim 2')

plt.tight_layout()
plt.savefig('./output/embedding_visualization.png')
print("✅ Visualization saved as ./output/embedding_visualization.png")
