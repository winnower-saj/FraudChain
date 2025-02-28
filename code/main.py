import sys
import os

from data_loader import load_data
from graph import build_graph
from train import train_model
from evaluate import evaluate_model
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

df_features = load_data()

pyg_data = build_graph(df_features)

model, train_idx, val_idx, test_idx = train_model(pyg_data, epochs=200)

evaluate_model(model, pyg_data, test_idx)

torch.save(model.state_dict(), "fraud_model.pth")
print("Model saved successfully!")
