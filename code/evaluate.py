import torch
from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, pyg_data, test_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    model.to(device)
    pyg_data = pyg_data.to(device)
    test_idx = test_idx.to(device)

    model.eval()
    with torch.no_grad():
        out = model(pyg_data)[test_idx]
        pred = out.argmax(dim=1)
        true_labels = pyg_data.y[test_idx]

        acc = (pred == true_labels).sum().item() / true_labels.size(0)
        auc = roc_auc_score(true_labels.cpu().numpy(), pred.cpu().numpy())

        print(f"Test Accuracy: {acc:.4f}, AUC-ROC: {auc:.4f}")
        print(classification_report(true_labels.cpu().numpy(), pred.cpu().numpy()))
