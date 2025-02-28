import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from model import FraudGNN

def train_model(pyg_data, epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pyg_data = pyg_data.to(device)

    train_idx, test_idx = train_test_split(range(pyg_data.num_nodes), test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=42)  # 10% validation

    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)

    model = FraudGNN(in_channels=pyg_data.x.shape[1], hidden_channels=64, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(pyg_data)[train_idx]
        loss = criterion(out, pyg_data.y[train_idx])

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(pyg_data)[val_idx]
            val_loss = criterion(val_out, pyg_data.y[val_idx])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_fraud_model.pth")

        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

    return model, train_idx.cpu(), val_idx.cpu(), test_idx.cpu()
