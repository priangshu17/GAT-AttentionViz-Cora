# GAT-AttentionViz-Cora

Implementation of **Graph Attention Networks (GAT)** on the **Cora citation network** using **PyTorch Geometric**, with a major focus on **visualizing learned attention scores**.

This project generates:

- 🎯 **Node classification** on Cora using a 2-layer GAT  
- 🔍 **Learned attention weights** extracted from each GATConv layer  
- 🌈 **Static graph visualizations** (attention-based edge width & color)  
- 📊 **Class→Class attention heatmap**  
- 👁 **Per-node neighbor attention diagrams**    

---

## 🚀 Features

### ✔ Train a multi-head GAT on Cora  
- 8 heads in the first layer  
- 1 head in the final output layer  
- Dropout, ELU, Adam optimizer  
- Full train/val/test accuracy reporting  

### ✔ Extract attention coefficients  
Using PyG's  
```python
GATConv(..., return_attention_weights=True)

