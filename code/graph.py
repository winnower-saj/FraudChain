import os
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
import pandas as pd
import numpy as np

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../elliptic_bitcoin_dataset/")

def build_graph(df_features):
    df_edges = pd.read_csv(os.path.join(DATASET_PATH, "elliptic_txs_edgelist.csv"), header=None, names=["source", "target"])

    G = nx.DiGraph() # directed graph

    for _, row in df_features.iterrows():
        G.add_node(row["tx_id"], features=row.iloc[:-2].values, label=row["class"])

    for _, row in df_edges.iterrows():  # add edge (transcation) between 2 nodes if they exist
        if row["source"] in G.nodes and row["target"] in G.nodes:
            G.add_edge(row["source"], row["target"])

    pyg_data = from_networkx(G) # converting to PyTorch Geometric format
    pyg_data.x = torch.tensor(np.array([G.nodes[n]["features"] for n in G.nodes]), dtype=torch.float)
    pyg_data.y = torch.tensor(np.array([G.nodes[n]["label"] for n in G.nodes]), dtype=torch.long)

    return pyg_data
