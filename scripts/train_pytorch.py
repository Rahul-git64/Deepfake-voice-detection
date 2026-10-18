import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def compute_eer(y_true, y_scores):
    """
    Computes Equal Error Rate (EER) and threshold using pure numpy / torch.
    EER occurs where False Acceptance Rate (FAR) == False Rejection Rate (FRR).
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    # Sort scores descending
    desc_order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_order]
    y_scores_sorted = y_scores[desc_order]

    # Calculate positives and negatives
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.5

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    fnr = 1.0 - (tps / n_pos)
    fpr = fps / n_neg

    diffs = np.abs(fnr - fpr)
    idx = np.argmin(diffs)
    eer = (fnr[idx] + fpr[idx]) / 2.0
    threshold = y_scores_sorted[idx]
    return float(eer), float(threshold)


class DeepfakeClassifier(nn.Module):
    def __init__(self, in_features=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


def load_dataset(pt_path: Path):
    if not pt_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {pt_path}")
    data = torch.load(pt_path, weights_only=False)
    X = data["embeddings"].float()
    y = data["labels"].long()
    filenames = data.get("filenames", [])
    return X, y, filenames


def evaluate(model, X, y, device, split_name="Dev"):
    model.eval()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(by.numpy())
            
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = np.mean(all_preds == all_targets)
    eer, threshold = compute_eer(all_targets, all_probs)
    
    print(f"\n--- {split_name} Evaluation ---", flush=True)
    print(f"Accuracy: {acc:.4f}", flush=True)
    print(f"EER:      {eer:.4f} (Threshold: {threshold:.4f})", flush=True)
    
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    tn = np.sum((all_preds == 0) & (all_targets == 0))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))
    print(f"Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]", flush=True)
    
    return acc, eer


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch Deepfake Audio Classifier on Wav2Vec2 Embeddings")
    parser.add_argument("--embeddings_dir", type=str, default="data/embeddings")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_model", type=str, default="models/classifier.pt")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)
    train_path = emb_dir / "train_embeddings.pt"
    dev_path = emb_dir / "dev_embeddings.pt"

    print("Loading extracted embeddings...", flush=True)
    X_train, y_train, _ = load_dataset(train_path)
    X_dev, y_dev, _ = load_dataset(dev_path)

    print(f"Train samples: {X_train.shape[0]} | Dev samples: {X_dev.shape[0]}", flush=True)
    print(f"Train labels: bonafide={(y_train == 0).sum().item()}, spoof={(y_train == 1).sum().item()}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}", flush=True)

    model = DeepfakeClassifier(in_features=768).to(device)

    # Compute class weights for class balance
    n_bonafide = (y_train == 0).sum().item()
    n_spoof = (y_train == 1).sum().item()
    total = len(y_train)
    weight = torch.tensor([total / (2.0 * n_bonafide), total / (2.0 * n_spoof)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    best_dev_eer = 1.0
    out_file = Path(args.output_model)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("\nStarting training loop...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)

        avg_loss = total_loss / len(train_dataset)
        acc, eer = evaluate(model, X_dev, y_dev, device, split_name=f"Epoch {epoch} Dev")
        print(f"Epoch {epoch:02d}/{args.epochs:02d} - Train Loss: {avg_loss:.4f} - Dev Acc: {acc:.4f} - Dev EER: {eer:.4f}", flush=True)

        if eer < best_dev_eer:
            best_dev_eer = eer
            torch.save({
                "model_state_dict": model.state_dict(),
                "in_features": 768,
                "best_eer": best_dev_eer,
                "epoch": epoch
            }, out_file)
            print(f"[BEST] Saved new best model (EER: {best_dev_eer:.4f}) to {out_file}", flush=True)

    print(f"\nTraining complete! Best Dev EER achieved: {best_dev_eer:.4f}", flush=True)


if __name__ == "__main__":
    main()
