# cddm_analysis_all.py
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import umap
from torch_geometric.nn import SAGEConv, GATConv

# -------- Paths -------- #
GRAPH_PATH = "cddm_graph.pt"
OUTPUT_DIR = "./log/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- Load Graph -------- #
print(f"Loading graph from {GRAPH_PATH}...")
graph = torch.load(GRAPH_PATH, weights_only=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
graph = graph.to(device)

labels = graph.y.detach().cpu().numpy()

# -------- Model Definitions (MATCH trainer names!) -------- #
class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGNN, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None, return_embeddings=False):
        x = self.lin1(x)
        x = F.relu(x)
        if return_embeddings:
            return x
        x = self.lin2(x)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_embeddings=False):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if return_embeddings:
            return x
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index, return_embeddings=False):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        if return_embeddings:
            return x
        x = self.conv2(x, edge_index)
        return x

# -------- Model registry (paths must match your trainer outputs) -------- #
num_features = graph.num_node_features
num_classes = int(graph.y.max().item() + 1)

MODELS = {
    "SimpleGNN": {
        "cls": SimpleGNN,
        "args": (num_features, 128, num_classes),
        "path": "./Result/SimpleGNN_final.pth",
    },
    "GraphSAGE": {
        "cls": GraphSAGE,
        "args": (num_features, 128, num_classes),
        "path": "./Result/GraphSAGE_final.pth",
    },
    "GAT": {
        "cls": GAT,
        "args": (num_features, 128, num_classes),
        "path": "./Result/GAT_final.pth",
    },
}

# -------- Utils -------- #
def safe_to_networkx(pg_graph, max_nodes_for_plot=2000):
    """Convert to networkx and (optionally) downsample for sane plotting."""
    G = to_networkx(pg_graph)
    if G.number_of_nodes() > max_nodes_for_plot:
        # simple downsampling: take first N nodes
        nodes = list(G.nodes())[:max_nodes_for_plot]
        G = G.subgraph(nodes).copy()
    return G

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# -------- Analytics for each model -------- #
def analyze_model(name, spec):
    model_path = spec["path"]
    out_dir = os.path.join(OUTPUT_DIR, name)
    ensure_dir(out_dir)

    if not os.path.exists(model_path):
        print(f"⚠️  Skipping {name}: checkpoint not found at {model_path}")
        return

    print(f"\n🔍 Analyzing {name}...")
    # Build & load model
    model = spec["cls"](*spec["args"]).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        # intermediate embeddings for UMAP
        emb = model(graph.x, graph.edge_index, return_embeddings=True).detach().cpu().numpy()

    # Choose eval split if available
    if hasattr(graph, "test_mask"):
        mask = graph.test_mask.detach().cpu().numpy()
        y_true = labels[mask]
        y_pred = preds[mask]
        split_used = "test"
    else:
        y_true = labels
        y_pred = preds
        split_used = "all-nodes (no test_mask)"

    acc = accuracy_score(y_true, y_pred)
    print(f"✅ {name} accuracy on {split_used}: {acc:.4f}")

    # 1) Graph Structure
    print("  • Saving graph_structure.png ...")
    G = safe_to_networkx(graph)
    plt.figure(figsize=(8, 8))
    try:
        nx.draw(
            G,
            node_size=8,
            node_color=labels[: G.number_of_nodes()],
            cmap="tab20",
            edge_color="lightgray",
            linewidths=0.0,
        )
    except Exception:
        # fallback plain draw
        nx.draw(G, node_size=8, edge_color="lightgray", linewidths=0.0)
    plt.title(f"{name} - Graph Structure")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "graph_structure.png"), dpi=200)
    plt.close()

    # 2) Class Distribution
    print("  • Saving class_distribution.png ...")
    unique, counts = torch.unique(graph.y.detach().cpu(), return_counts=True)
    plt.figure(figsize=(6, 4))
    plt.bar(unique.numpy(), counts.numpy())
    plt.title(f"{name} - Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=200)
    plt.close()

    # 3) Confusion Matrix
    print("  • Saving confusion_matrix.png ...")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="viridis", values_format="d", colorbar=False)
    plt.title(f"{name} - Confusion Matrix ({split_used})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # 4) UMAP Embedding
    print("  • Saving umap_embedding.png ...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, verbose=False)
    umap_result = reducer.fit_transform(emb)
    plt.figure(figsize=(6, 6))
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, s=6, cmap="tab20")
    plt.title(f"{name} - UMAP Embedding")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "umap_embedding.png"), dpi=200)
    plt.close()

    # Summary file
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Checkpoint: {model_path}\n")
        f.write(f"Split evaluated: {split_used}\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write("Artifacts:\n")
        f.write("  - graph_structure.png\n")
        f.write("  - class_distribution.png\n")
        f.write("  - confusion_matrix.png\n")
        f.write("  - umap_embedding.png\n")

# -------- Run -------- #
for model_name, spec in MODELS.items():
    analyze_model(model_name, spec)

print(f"\n✅ All analytics saved under: {OUTPUT_DIR}")
