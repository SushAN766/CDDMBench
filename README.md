# A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis

## Introduction
CDDM dataset is the crop disease domain multimodal dataset, a pioneering resource designed to advance the field of agricultural research through the application of multimodal learning techniques. 


## CDDM dataset
The CDDM dataset includes images and conversation data. 
### CDDM images:
Please download CDDM images from the following link and extract it to the /dataset/ directory.
- [Google Drive](https://drive.google.com/file/d/1kfB3zkittoef4BasOhwvAb8Cb66EPXst/view?usp=sharing)
- [Baidu Yun Pan](https://pan.baidu.com/s/1CgmO2MyEKV6EE42eNS0sIw?pwd=ip1r): ip1r


### CDDM conversation:
We offer the conversation data in two formats suitable for training Qwen-VL and LLaVA models. The data covers crop disease diagnosis and knowledge.

Please extract the conversation data to the /dataset/VQA/ directory. 
- [Qwen-VL training data](dataset/VQA/Crop_Disease_train_qwenvl.zip)
- [LLaVA training data](dataset/VQA/Crop_Disease_train_llava.zip)
- [Test data](dataset/VQA/test_dataset.zip)

## Train
### Qwen-VL: To run on a machine with 8 GPUs:
```shell
cd Qwen-VL
sh finetune/finetune_lora_ds.sh
```
## Test
### Measure the accuracy of disease diagnosis
Generate Diagnosis_output.json by performing inference on the test data with the model. Afterward, execute the following code.
```shell
cd script
python disease_diagnosis_accuracy.py
```

## Citation
If you find our dataset or model useful, please cite our work:

```bibtex
@InProceedings{10.1007/978-3-031-73016-0_10,
author="Xiang, Liu
and Zhaoxiang, Liu
and Huan, Hu
and Zezhou, Chen
and Kohou, Wang
and Kai, Wang
and Shiguo, Lian",
title="A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis",
booktitle="Computer Vision -- ECCV 2024",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="157--170",
isbn="978-3-031-73016-0"
}
```
## Paper
For more details, please refer to our paper: [ECCV 2024 Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11606.pdf)  , [arxiv](https://arxiv.org/abs/2503.06973)

---
# Analysis of Multimodal Data Using GNN (CDDM Dataset)

This project demonstrates how to process multimodal data (images + text) from the **CDDM dataset**, convert it into graph format, and train a **Graph Neural Network (GNN)** to perform classification using PyTorch Geometric.

---

## 📁 Folder Structure
```plaintext
multimodal-gnn-cddm/                   
├── dataset/images/                    # CDDM image files
├── gnn-env/                           # Python virtual environment (excluded via .gitignore)
├── VQA/.json zip files                             # Visual Question Answering components
├── .gitignore                         # Git ignore rules
├── Crop_Disease_train_llava.json     # CDDM conversation file (LLaVA format)
├── Crop_Disease_train_qwenvl.json    # CDDM conversation file (Qwen-VL format)
├── disease_knowledge.json            # Domain-specific knowledge base
├── disease_diagnosis.json            # Ground-truth diagnosis labels
├── cddm_graph_builder.py             # Feature extraction + graph construction script
├── cddm_gnn_trainer.py               # GCN model training script
├── cddm_visualizer.py                # Training & embedding visualization (PCA/t-SNE)
├── ccdm_graph.pt                 # Saved PyG graph
├── output/                           # Output directory (auto-created)
│   
│   ├── gnn_model.pth                 # Trained model weights
│   ├── training_log.txt              # Training logs (optional)
│   ├── embedding_visualization.png  # Visual representation of embeddings
│   └── loss_accuracy_plot.png       # Training loss & accuracy plot
└── README.md                         # Project documentation


```
---

## 🔧 Setup Instructions

### ✅ Step 1: Create Virtual Environment (Optional)

```bash
python -m venv gnn-env
source gnn-env/bin/activate        # Linux/macOS
gnn-env\Scripts\activate          # Windows
```
## ✅ Step 2: Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.1+cu117.html  # Replace cu117 if needed
pip install transformers
pip install scikit-learn
pip install matplotlib
pip install tqdm
pip install git+https://github.com/openai/CLIP.git
pip install umap-learn networkx scikit-learn matplotlib
```
---
## 🚀 How to Run

### 📌 Step 1: Graph Construction

```shell
python cddm_graph_builder.py
```

- Loads Crop_Disease_train_llava.json and image folder
- Extracts image features using ResNet50
- Extracts text features using BERT
- Constructs a graph and saves it as output/graph_data.pt

### 📌 Step 2: Train GCN Model

```shell
python cddm_gnn_trainer.py
```
- Loads the graph
- Trains a 2-layer GCN for classification
- Saves model as output/gnn_model.pth

### 📌 Step 3: Visualize Results

```shell
python cddm_visualizer.py
```
- Reduces node embeddings using t-SNE or PCA
- Generates: embedding_visualization.png
- If training_log.txt exists, also plots loss_accuracy_plot.png


### 📌 Step 4: Analytics & Extra Plots

```bash
python cddm_analysis.py
```
✅ This will create the following files in the ./output folder:

- graph_structure.png – Visualization of the graph connectivity

- class_distribution.png – Distribution of disease categories

- confusion_matrix.png – Model predictions vs actual labels

- umap_embedding.png – UMAP visualization of node embeddings

---

## 🎯 Project Objectives

- Preprocess multimodal data (image + text)  
- Convert preprocessed data into graph representation  
- Apply GNN models for prediction and insight  

---

## 📈 Possible Improvements

- Use **GraphSAGE** or **GAT** instead of GCN  
- Construct edges based on **cosine similarity** or **KNN**  
- Use **multi-task** or **multi-label** outputs  

---

## 📬 Credits

- **Dataset**: Crop Disease Diagnosis Multimodal (CDDM)  
- **Libraries**: PyTorch Geometric,HuggingFace Transformers,scikit-learn,matplotlib
