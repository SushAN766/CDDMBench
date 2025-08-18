# cddm_gnn_trainer.py (SimpleGNN, GraphSAGE, GAT)
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
import matplotlib.pyplot as plt
import os

# -------- LOAD GRAPH -------- #
GRAPH_PATH = "cddm_graph.pt"
print(f"Loading graph from {GRAPH_PATH}...")
graph = torch.load(GRAPH_PATH, weights_only=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
graph = graph.to(device)

# Ensure output folder exists
os.makedirs("./Result", exist_ok=True)

# -------- TRAIN/TEST SPLIT -------- #
if not hasattr(graph, 'train_mask'):
    print("⚠️ train_mask or test_mask not found. Creating them now...")
    num_nodes = graph.num_nodes
    perm = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    train_idx, test_idx = perm[:train_size], perm[train_size:]

    graph.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph.train_mask[train_idx] = True
    graph.test_mask[test_idx] = True

# -------- MODEL DEFINITIONS -------- #
class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGNN, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x


# -------- TRAINING FUNCTION -------- #
def train_model(model_name, model_class):
    model = model_class(graph.num_node_features, 128, graph.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    best_acc = 0
    patience = 10
    wait = 0
    loss_history = []
    acc_history = []

    print(f"🚀 Starting training: {model_name}...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = F.cross_entropy(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = out.argmax(dim=1)
        correct = pred[graph.test_mask] == graph.y[graph.test_mask]
        acc = int(correct.sum()) / int(graph.test_mask.sum())

        loss_history.append(loss.item())
        acc_history.append(acc)

        print(f"{model_name} | Epoch {epoch:03d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            wait = 0
            torch.save(model.state_dict(), f"./output/{model_name}_best.pth")
        else:
            wait += 1
            if wait >= patience:
                print("⏹ Early stopping triggered!")
                break

    torch.save(model.state_dict(), f"./Result/{model_name}_final.pth")
    print(f"✅ Training complete for {model_name}! Best Accuracy: {best_acc:.4f}")

    # -------- PLOT LOSS & ACCURACY -------- #
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title(f'{model_name} - Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc_history, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./Result/{model_name}_loss_accuracy_plot.png')
    print(f"📊 Training curve saved as ./Result/{model_name}_loss_accuracy_plot.png")


# -------- TRAIN ALL MODELS -------- #
train_model("SimpleGNN", SimpleGNN)
train_model("GraphSAGE", GraphSAGE)
train_model("GAT", GAT)
