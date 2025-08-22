# cddm_analysis.py - GPU Optimized Version
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import umap # type: ignore
from torch_geometric.nn import GCNConv
import os
import numpy as np
from tqdm import tqdm

# Enable GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Paths
GRAPH_PATH = "cddm_graph.pt"
MODEL_PATH = "./output/gnn_model_best.pth"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check GPU availability and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# -------- Load Graph -------- #
print(f"Loading graph from {GRAPH_PATH}...")
graph = torch.load(GRAPH_PATH, weights_only=False)
graph = graph.to(device)

# Keep labels on GPU until needed for specific operations
labels_gpu = graph.y
labels_cpu = labels_gpu.cpu().numpy()  # Only convert once when needed

print(f"Graph loaded: {graph.num_nodes} nodes, {graph.num_edges} edges")
print(f"Node features: {graph.num_node_features}, Classes: {graph.y.max().item() + 1}")

# -------- Define GCN Model (same as training) -------- #
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_embeddings: bool = False):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        if return_embeddings:
            return x
        x = self.conv2(x, edge_index)
        return x

# Load model with optimizations
print("Loading and optimizing model...")
model = GCN(graph.num_node_features, 128, graph.y.max().item() + 1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# Enable inference optimizations
if device == 'cuda':
    try:
        model = torch.jit.script(model)  # JIT compilation for faster inference
        print("JIT compilation successful")
    except Exception as e:
        print(f"JIT compilation failed, using regular model: {e}")
        # Continue with regular model if JIT fails
    
# -------- Get Predictions & Embeddings -------- #
print("Generating predictions and embeddings...")
with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device=='cuda')):  # Mixed precision
    # Get predictions
    out = model(graph.x, graph.edge_index)
    pred_gpu = out.argmax(dim=1)
    pred_cpu = pred_gpu.cpu().numpy()
    
    # Get embeddings
    embeddings_gpu = model(graph.x, graph.edge_index, return_embeddings=True)
    
    # Clear cache to free up GPU memory
    if device == 'cuda':
        torch.cuda.empty_cache()

print(f"Model inference completed on {device}")

# -------- 1. Graph Structure Visualization (Sampled for large graphs) -------- #
print("Generating graph structure visualization...")
if graph.num_nodes > 5000:
    print(f"Large graph detected ({graph.num_nodes} nodes). Sampling 2000 nodes for visualization...")
    # Sample nodes for visualization - ensure proper device handling
    sample_indices = torch.randperm(graph.num_nodes, device='cpu')[:2000]  # Generate on CPU
    
    # Move graph to CPU for subgraph operation
    graph_cpu = graph.cpu()
    subgraph_data = graph_cpu.subgraph(sample_indices)
    
    # Convert to NetworkX
    G = to_networkx(subgraph_data.to('cpu'))
    sample_labels = labels_cpu[sample_indices.numpy()]
    node_colors = sample_labels
    
    # Move original graph back to GPU
    graph = graph.to(device)
else:
    # For smaller graphs, move to CPU for NetworkX conversion
    graph_cpu = graph.cpu()
    G = to_networkx(graph_cpu)
    node_colors = labels_cpu
    # Move back to GPU
    graph = graph.to(device)

plt.figure(figsize=(10, 10))
nx.draw(G, node_size=15, node_color=node_colors, cmap='tab20', edge_color='gray', alpha=0.7)
plt.title("Graph Structure")
plt.savefig(f"{OUTPUT_DIR}/graph_structure.png", dpi=150, bbox_inches='tight')
plt.close()

# -------- 2. Class Distribution -------- #
print("Generating class distribution chart...")
unique_labels, counts = torch.unique(labels_gpu, return_counts=True)
unique_labels_cpu = unique_labels.cpu().numpy()
counts_cpu = counts.cpu().numpy()

plt.figure(figsize=(10, 6))
bars = plt.bar(unique_labels_cpu, counts_cpu, alpha=0.8, edgecolor='black')
plt.title("Class Distribution", fontsize=16)
plt.xlabel("Class", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(unique_labels_cpu)

# Add value labels on bars
for bar, count in zip(bars, counts_cpu):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01*max(counts_cpu),
             f'{count}', ha='center', va='bottom', fontsize=10)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# -------- 3. Confusion Matrix -------- #
print("Generating confusion matrix...")
cm = confusion_matrix(labels_cpu, pred_cpu)
disp = ConfusionMatrixDisplay(cm, display_labels=unique_labels_cpu)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap='Blues', ax=ax, colorbar=True)
plt.title("Confusion Matrix", fontsize=16)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

# -------- Calculate Classification Metrics -------- #
print("Computing classification metrics...")

# Overall Accuracy
accuracy = np.sum(pred_cpu == labels_cpu) / len(labels_cpu)

# Per-class metrics
num_classes = len(unique_labels_cpu)
class_metrics = {}

for class_idx in unique_labels_cpu:
    # True positives, false positives, true negatives, false negatives
    tp = np.sum((labels_cpu == class_idx) & (pred_cpu == class_idx))
    fp = np.sum((labels_cpu != class_idx) & (pred_cpu == class_idx))
    tn = np.sum((labels_cpu != class_idx) & (pred_cpu != class_idx))
    fn = np.sum((labels_cpu == class_idx) & (pred_cpu != class_idx))
    
    # Sensitivity (Recall/True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Class-specific accuracy
    class_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # Store metrics
    class_metrics[class_idx] = {
        'accuracy': class_accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'support': tp + fn  # Number of true instances for this class
    }

# Calculate macro averages
macro_accuracy = np.mean([metrics['accuracy'] for metrics in class_metrics.values()])
macro_sensitivity = np.mean([metrics['sensitivity'] for metrics in class_metrics.values()])
macro_specificity = np.mean([metrics['specificity'] for metrics in class_metrics.values()])

print(f"\n📊 Classification Performance Metrics:")
print(f"Overall Accuracy (ACC): {accuracy:.4f}")
print(f"Macro-averaged Sensitivity (SEN): {macro_sensitivity:.4f}")
print(f"Macro-averaged Specificity (SPE): {macro_specificity:.4f}")
print(f"Subject Number: {len(labels_cpu):,}")

# Per-class detailed metrics
print(f"\n📋 Per-Class Metrics:")
print(f"{'Class':<6} {'ACC':<8} {'SEN':<8} {'SPE':<8} {'Support':<8}")
print("-" * 40)
for class_idx in sorted(unique_labels_cpu):
    metrics = class_metrics[class_idx]
    print(f"{class_idx:<6} {metrics['accuracy']:<8.4f} {metrics['sensitivity']:<8.4f} "
          f"{metrics['specificity']:<8.4f} {metrics['support']:<8}")

print(f"\nMacro Avg: {macro_accuracy:<6.4f} {macro_sensitivity:<8.4f} {macro_specificity:<8.4f} {len(labels_cpu):<8}")

# -------- 6. Metrics Visualization -------- #
print("Generating classification metrics visualization...")

# Create metrics comparison chart
metrics_data = {
    'Class': list(unique_labels_cpu),
    'Accuracy': [class_metrics[cls]['accuracy'] for cls in unique_labels_cpu],
    'Sensitivity': [class_metrics[cls]['sensitivity'] for cls in unique_labels_cpu],
    'Specificity': [class_metrics[cls]['specificity'] for cls in unique_labels_cpu]
}

# Metrics bar plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Individual metrics per class
x_pos = np.arange(len(unique_labels_cpu))
width = 0.25

bars1 = ax1.bar(x_pos - width, metrics_data['Accuracy'], width, label='Accuracy', alpha=0.8, color='skyblue')
bars2 = ax1.bar(x_pos, metrics_data['Sensitivity'], width, label='Sensitivity', alpha=0.8, color='lightcoral')
bars3 = ax1.bar(x_pos + width, metrics_data['Specificity'], width, label='Specificity', alpha=0.8, color='lightgreen')

ax1.set_xlabel('Class', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Per-Class Classification Metrics', fontsize=14)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'Class {cls}' for cls in unique_labels_cpu])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 1.1)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Overall metrics summary
overall_metrics = ['Overall ACC', 'Macro SEN', 'Macro SPE']
overall_values = [accuracy, macro_sensitivity, macro_specificity]
colors = ['skyblue', 'lightcoral', 'lightgreen']

bars = ax2.bar(overall_metrics, overall_values, color=colors, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Overall Classification Performance', fontsize=14)
ax2.set_ylim(0, 1.1)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, value in zip(bars, overall_values):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/classification_metrics.png", dpi=150, bbox_inches='tight')
plt.close()

# -------- 7. Confusion Matrix with Metrics -------- #
print("Generating enhanced confusion matrix with metrics...")

# Enhanced confusion matrix with normalized version
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw confusion matrix
disp1 = ConfusionMatrixDisplay(cm, display_labels=[f'Class {i}' for i in unique_labels_cpu])
disp1.plot(cmap='Blues', ax=axes[0], colorbar=True, values_format='d')
axes[0].set_title(f'Confusion Matrix (Raw Counts)\nOverall Accuracy: {accuracy:.4f}', fontsize=14)

# Normalized confusion matrix
cm_normalized = confusion_matrix(labels_cpu, pred_cpu, normalize='true')
disp2 = ConfusionMatrixDisplay(cm_normalized, display_labels=[f'Class {i}' for i in unique_labels_cpu])
disp2.plot(cmap='Blues', ax=axes[1], colorbar=True, values_format='.3f')
axes[1].set_title(f'Confusion Matrix (Normalized)\nMacro SEN: {macro_sensitivity:.4f}, Macro SPE: {macro_specificity:.4f}', fontsize=14)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/enhanced_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

# -------- 8. Save Metrics to File -------- #
print("Saving detailed metrics to file...")

# Create detailed metrics report
metrics_report = []
metrics_report.append("=" * 60)
metrics_report.append("CLASSIFICATION PERFORMANCE REPORT")
metrics_report.append("=" * 60)
metrics_report.append(f"Dataset: {GRAPH_PATH}")
metrics_report.append(f"Model: {MODEL_PATH}")
metrics_report.append(f"Device: {device}")
metrics_report.append(f"Subject Number: {len(labels_cpu):,}")
metrics_report.append("")

metrics_report.append("OVERALL METRICS:")
metrics_report.append(f"  Overall Accuracy (ACC): {accuracy:.6f}")
metrics_report.append(f"  Macro-averaged Sensitivity (SEN): {macro_sensitivity:.6f}")
metrics_report.append(f"  Macro-averaged Specificity (SPE): {macro_specificity:.6f}")
metrics_report.append("")

metrics_report.append("PER-CLASS DETAILED METRICS:")
metrics_report.append(f"{'Class':<8} {'ACC':<10} {'SEN':<10} {'SPE':<10} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Support':<8}")
metrics_report.append("-" * 80)

for class_idx in sorted(unique_labels_cpu):
    m = class_metrics[class_idx]
    metrics_report.append(f"{class_idx:<8} {m['accuracy']:<10.6f} {m['sensitivity']:<10.6f} "
                         f"{m['specificity']:<10.6f} {m['tp']:<6} {m['fp']:<6} "
                         f"{m['tn']:<6} {m['fn']:<6} {m['support']:<8}")

metrics_report.append("")
metrics_report.append(f"{'Macro':<8} {macro_accuracy:<10.6f} {macro_sensitivity:<10.6f} {macro_specificity:<10.6f}")
metrics_report.append("")

# Class distribution
metrics_report.append("CLASS DISTRIBUTION:")
for class_idx in sorted(unique_labels_cpu):
    count = counts_cpu[unique_labels_cpu == class_idx][0]
    percentage = (count / len(labels_cpu)) * 100
    metrics_report.append(f"  Class {class_idx}: {count:,} samples ({percentage:.2f}%)")

# Save report
with open(f"{OUTPUT_DIR}/classification_metrics_report.txt", 'w') as f:
    f.write('\n'.join(metrics_report))

print(f"Detailed metrics report saved to: {OUTPUT_DIR}/classification_metrics_report.txt")
print("Generating UMAP embedding visualization...")

# Convert embeddings to CPU for UMAP (but optimize the process)
embeddings_cpu = embeddings_gpu.cpu().numpy()

# Clear GPU memory
if device == 'cuda':
    del embeddings_gpu
    torch.cuda.empty_cache()

# Use optimized UMAP parameters for speed
print("Computing UMAP projection...")
reducer = umap.UMAP(
    n_neighbors=15, 
    min_dist=0.1, 
    n_components=2,
    metric='cosine',
    n_jobs=-1,  # Use all CPU cores
    random_state=42,
    verbose=True
)

umap_result = reducer.fit_transform(embeddings_cpu)

# Create UMAP visualization
plt.figure(figsize=(12, 10))
scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], 
                     c=labels_cpu, cmap='tab20', s=8, alpha=0.7, edgecolors='none')
plt.title("UMAP Embedding Visualization", fontsize=16)
plt.xlabel("UMAP 1", fontsize=14)
plt.ylabel("UMAP 2", fontsize=14)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Class', fontsize=12)
cbar.set_ticks(unique_labels_cpu)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/umap_embedding.png", dpi=150, bbox_inches='tight')
plt.close()

# -------- 10. Additional GPU-based Analysis -------- #
print("Computing additional GPU-based metrics...")

# Node degree analysis (GPU accelerated)
with torch.no_grad():
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    
    # Compute node degrees on GPU
    row, col = edge_index
    degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
    degree.scatter_add_(0, row, torch.ones_like(row))
    degree_cpu = degree.cpu().numpy()

# Degree distribution plot
plt.figure(figsize=(10, 6))
plt.hist(degree_cpu, bins=50, alpha=0.7, edgecolor='black')
plt.title("Node Degree Distribution", fontsize=16)
plt.xlabel("Degree", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/degree_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# -------- Performance Summary -------- #
print(f"\n✅ All analytics saved in {OUTPUT_DIR}")
print(f"📊 Analysis Summary:")
print(f"   - Subject Number: {len(labels_cpu):,}")
print(f"   - Nodes: {graph.num_nodes:,}")
print(f"   - Edges: {graph.num_edges:,}")
print(f"   - Classes: {len(unique_labels_cpu)}")
print(f"   - Overall Accuracy (ACC): {accuracy:.4f}")
print(f"   - Macro Sensitivity (SEN): {macro_sensitivity:.4f}")
print(f"   - Macro Specificity (SPE): {macro_specificity:.4f}")
print(f"   - Average Node Degree: {degree_cpu.mean():.2f}")
print(f"   - Device Used: {device}")

if device == 'cuda':
    print(f"   - Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    torch.cuda.empty_cache()  # Final cleanup

# Summary of generated files
generated_files = [
    "graph_structure.png",
    "class_distribution.png", 
    "enhanced_confusion_matrix.png",
    "classification_metrics.png",
    "umap_embedding.png",
    "degree_distribution.png",
    "classification_metrics_report.txt"
]

print(f"\n📁 Generated Files:")
for file in generated_files:
    print(f"   - {OUTPUT_DIR}/{file}")