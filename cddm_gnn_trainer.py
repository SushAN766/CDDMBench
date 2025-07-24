# cddm_gnn_trainer.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import os

# -------- LOAD GRAPH -------- #
GRAPH_PATH = "cddm_graph.pt"
print(f"Loading graph from {GRAPH_PATH}...")
graph = torch.load(GRAPH_PATH, weights_only=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
graph = graph.to(device)

# Ensure output folder exists
os.makedirs("./output", exist_ok=True)

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

# -------- CLASS NAMES -------- #
class_names = [f"Class {i}" for i in range(graph.y.max().item() + 1)]

# -------- DEFINE GCN MODEL -------- #
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

model = GCN(graph.num_node_features, 128, graph.y.max().item() + 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------- TRAINING -------- #
epochs = 100
best_acc = 0
patience = 10
wait = 0
loss_history = []
acc_history = []

print("Starting training...")
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

    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        wait = 0
        torch.save(model.state_dict(), "./output/gnn_model_best.pth")
    else:
        wait += 1
        if wait >= patience:
            print("⏹ Early stopping triggered!")
            break

# -------- SAVE FINAL MODEL -------- #
torch.save(model.state_dict(), "./output/gnn_model_final.pth")
print(f"✅ Training complete! Best Accuracy: {best_acc:.4f}")
print("✅ Models saved: gnn_model_best.pth & gnn_model_final.pth")

# -------- PLOT LOSS & ACCURACY -------- #
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_history, label='Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('./output/loss_accuracy_plot.png')
print("📊 Training curve saved as ./output/loss_accuracy_plot.png")
