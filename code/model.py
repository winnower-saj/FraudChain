import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, GATConv

class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(FraudGNN, self).__init__()
        self.conv1 = GraphSAGE(in_channels, hidden_channels, num_layers=num_layers) # embeddings for transcation
        self.conv2 = GATConv(hidden_channels, out_channels) # weigh the importance of different transcations

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
