# cddm_gnn_visualizer.py
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# -------- CONFIG -------- #
GRAPH_PATH = "cddm_graph.pt"
OUTPUT_DIR = "./Outcome/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------- LOAD GRAPH -------- #
print(f"📂 Loading graph from {GRAPH_PATH}...")
graph = torch.load(GRAPH_PATH, weights_only=False)
graph = graph.to(device)

# -------- MODELS -------- #
from torch_geometric.nn import SAGEConv, GATConv

class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGNN, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# -------- VISUALIZATION FUNCTION -------- #
def visualize_embeddings(model_name, model_class, weight_path):
    print(f"🎨 Visualizing embeddings for {model_name}...")
    model = model_class(graph.num_node_features, 128, graph.y.max().item() + 1).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        embeddings = model(graph.x, graph.edge_index).cpu()

    labels = graph.y.cpu().numpy()

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    plt.figure(figsize=(6, 5))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap="tab20", s=10)
    plt.title(f"{model_name} - PCA")
    plt.colorbar()
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_pca.png")
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca')
    tsne_result = tsne.fit_transform(embeddings)
    plt.figure(figsize=(6, 5))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap="tab20", s=10)
    plt.title(f"{model_name} - t-SNE")
    plt.colorbar()
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_tsne.png")
    plt.close()

    print(f"✅ Saved PCA and t-SNE plots for {model_name} in {OUTPUT_DIR}")

# -------- RUN FOR ALL MODELS -------- #
visualize_embeddings("SimpleGNN", SimpleGNN, "./Result/SimpleGNN_final.pth")
visualize_embeddings("GraphSAGE", GraphSAGE, "./Result/GraphSAGE_final.pth")
visualize_embeddings("GAT", GAT, "./Result/GAT_final.pth")
