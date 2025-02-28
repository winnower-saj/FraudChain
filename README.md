# FraudChain: Graph-Based Bitcoin Fraud Detection

FraudChain is a **Graph Neural Network (GNN)-powered fraud detection system** for Bitcoin transactions. It models the flow of digital currency as a graph and applies **GraphSAGE & GAT** to identify fraudulent activities.

## Features
- **Graph-Based Fraud Detection** : Models Bitcoin transactions as a graph and detects suspicious patterns.
- **Deep Learning on Blockchain Data** : Uses GraphSAGE & GAT for transaction classification.
- **Real-World Bitcoin Dataset**: Trained on the Elliptic Bitcoin Transaction dataset (200K+ transactions).
- **GPU-Optimized**: Supports CUDA for fast training and inference.
- **Fraud Prediction**: Classifies new transactions as fraudulent or legitimate.

---

## Dataset: Elliptic Bitcoin Transactions
FraudChain is trained on the **Elliptic Bitcoin Transaction Dataset**, which consists of:
- **200K+ transactions** forming a directed graph.
- **166 features per transaction**, including timestamps, amounts, and network behavior.
- **Fraud labels**:
  - `1 = Fraudulent`
  - `0 = Legitimate`
  - `Unknown = Removed from training`

**[Download Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-bitcoin-transactions)**

---

## Evaluation Metrics
FraudChain is evaluated using the following metrics:

| Metric       | Score  |
|-------------|--------|
| **Test Accuracy** | **97.91%** |
| **AUC-ROC** | **93.13%** |
| **Precision (Fraudulent)** | 91% |
| **Recall (Fraudulent)** | 87% |
| **F1-Score (Fraudulent)** | 89% 

**AUC-ROC** ensures the model **effectively distinguishes fraud from legitimate transactions**.  
**High recall** is crucial for fraud detection to minimize false negatives.  

