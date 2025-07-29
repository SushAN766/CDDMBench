# 📌 JSON Files Description for Multimodal Crop Disease Dataset

This section describes the structure and purpose of JSON files used in the project **Analysis of Multimodal Data Using Graph Neural Networks (GNN)**. These files provide both visual and textual information for creating multimodal node features.

---

## ✅ 1. `Crop_Disease_train_llava.json`

**Purpose:**  
Stores conversational Q&A pairs about crop diseases in a structured format, often used for Visual Question Answering (VQA) or instruction-based models.

### **Structure:**
```json
[
  {
    "id": "conv_34293_0",
    "image": "/dataset/images/Apple,Alternaria Blotch/plant_74609.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nWhat is the content of this picture?"},
      {"from": "gpt", "value": "This image shows an apple leaf affected by Alternaria Blotch."}
    ]
  }
]
```
### 🔍 Fields Description
- **id** → Unique identifier for the conversation  
- **image** → Path to the crop image  
- **conversations** → Dialogue turns between user (human) and model (gpt)  

---

### ✅ Usage in Project
- **Extract image path** → Used for **image feature extraction**  
- **Extract textual description from conversations** → Used for **multimodal feature creation**  

## ✅ 2. `Crop_Disease_train_qwenvl.json`
**Purpose:**
Contains questions and answers about crop images for a different model (QwenVL). Similar to LLaVA but formatted differently.

###  **Structure:**
```json
[
  {
    "id": "conv_46971_0",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>/dataset/images/Corn,Healthy/plant_138220.jpg</img>\nDescribe the content of this picture."
      },
      {"from": "assistant", "value": "This image shows a healthy corn leaf."}
    ]
  }
]
```
### 🔍 Fields Description
- **id** → Conversation ID  
- **conversations** → Dialogue between human and model (contains text and image reference)  

---

### ✅ Usage in Project
- **Extract image path** → From `<img>...</img>` tags within `conversations`  
- **Use Q&A text** → To enrich node features for **multimodal GNN**  

## ✅ 3. disease_diagnosis.json
**Purpose:**
Provides diagnostic Q&A pairs related to crop disease identification.

### **Structure:**
```json
[
  {
    "question_id": "test_conv_0001",
    "question": "Is this leaf from a grapevine?",
    "image": "/dataset/images/Apple,Leaf Rust/plant_98574.jpg",
    "answer": "No, this is an apple leaf."
  }
]
```
### 🔍 Fields Description
- **question_id** → Unique ID for the question  
- **question** → Diagnostic question  
- **image** → Path to the crop image  
- **answer** → Diagnosis or identification  

---

### ✅ Usage in Project
- **Combine `question` and `answer`** → Create a single **text feature**  
- **Link with `image`** → Build **multimodal representation** (image + text) for GNN  

## ✅ 4. disease_knowledge.json
**Purpose:**
Contains detailed knowledge-based questions and answers, e.g., symptoms, prevention, control methods.

### **Structure:**
```json
[
  {
    "question_id": "test_knowledge_conv_0001",
    "image": "/dataset/images/Rice,Blast/plant_121715.jpg",
    "question": "What measures can be taken to control Rice Blast?",
    "answer": "(1) Select resistant varieties ... (6) Rotate fungicides ..."
  }
]
```
### 🔍 Fields Description
- **question_id** → Knowledge entry ID  
- **image** → Path to crop image  
- **question** → Knowledge question  
- **answer** → Detailed answer (long text)  

---

### ✅ Usage in Project
- **Use `question` + `answer`** → Extract **semantic text features**  
- **Add domain knowledge** → Enrich node representation in the graph  
