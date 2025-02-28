import torch
import numpy as np
from model import FraudGNN
from graph import build_graph
from data_loader import load_data
from torch_geometric.data import Data

df_features = load_data()
pyg_data = build_graph(df_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on: {device}")

model = FraudGNN(in_channels=pyg_data.x.shape[1], hidden_channels=64, out_channels=2, num_layers=2).to(device)
model.load_state_dict(torch.load("best_fraud_model.pth", map_location=device))
model.eval()

def predict_fraud(new_transaction_features):
    transaction_tensor = torch.tensor(new_transaction_features, dtype=torch.float).unsqueeze(0).to(device)
    transaction_graph = Data(x=transaction_tensor, edge_index=torch.empty((2, 0), dtype=torch.long).to(device)) # sample empty transaction graph

    with torch.no_grad():
        output = model(transaction_graph)
        prediction = output.argmax(dim=1).item()

    return "Fraudulent" if prediction == 1 else "Legitimate"

example_transaction = np.random.rand(pyg_data.x.shape[1])  # sample data; can be replaced with new, unseen data

fraud_status = predict_fraud(example_transaction)
print(f"Predicted Status: {fraud_status}")
