import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from benchmarks.utils import (
    DeepfakeClassifier,
    compute_eer,
    DEVICE,
    RESULTS_DIR,
)

SEEDS = [42, 43, 44, 45, 46]

def evaluate_seed_model(model, X, y):
    model.eval()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_probs = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(DEVICE)
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
    eer, threshold, _, _, _ = compute_eer(all_targets, all_probs)
    return float(acc), float(eer), float(threshold)

def run_t5(quick: bool = False, full: bool = False):
    print("=" * 60)
    print("Running T5: Seed Variance Retraining (5 Seeds)")
    print("=" * 60)

    train_data = torch.load("data/embeddings/train_embeddings.pt", weights_only=False)
    dev_data = torch.load("data/embeddings/dev_embeddings.pt", weights_only=False)

    X_train = train_data["embeddings"].float()
    y_train = train_data["labels"].long()

    X_dev = dev_data["embeddings"].float()
    y_dev = dev_data["labels"].long()

    epochs = 3 if quick else 10
    batch_size = 128
    lr = 1e-3
    weight_decay = 1e-4

    n_bonafide = (y_train == 0).sum().item()
    n_spoof = (y_train == 1).sum().item()
    total = len(y_train)
    weights = torch.tensor([total / (2.0 * n_bonafide), total / (2.0 * n_spoof)]).to(DEVICE)

    save_dir = RESULTS_DIR / "seed_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for seed in SEEDS:
        print(f"\n--- Training Seed {seed} ({epochs} epochs) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = DeepfakeClassifier(in_features=768).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for ep in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(by)

            avg_loss = total_loss / len(train_dataset)

        # Final evaluation on dev split
        acc, eer, th = evaluate_seed_model(model, X_dev, y_dev)
        print(f"Seed {seed} Result: Dev Accuracy: {acc*100:.2f}%, Dev EER: {eer*100:.3f}% (threshold: {th:.4f})")

        # Save model without touching models/classifier.pt
        seed_model_path = save_dir / f"classifier_seed_{seed}.pt"
        torch.save({
            "seed": seed,
            "in_features": 768,
            "model_state_dict": model.state_dict(),
            "dev_acc": acc,
            "dev_eer": eer,
            "threshold": th,
        }, seed_model_path)

        results.append({
            "seed": seed,
            "dev_accuracy": acc,
            "dev_eer": eer,
            "eer_threshold": th,
            "model_path": str(seed_model_path),
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / "t5_seed_variance.csv", index=False)
    print(f"\nSeed variance results saved to {RESULTS_DIR / 't5_seed_variance.csv'}")

    mean_acc = res_df["dev_accuracy"].mean()
    std_acc = res_df["dev_accuracy"].std()
    mean_eer = res_df["dev_eer"].mean()
    std_eer = res_df["dev_eer"].std()

    print("\n--- Summary Across 5 Seeds ---")
    print(f"Dev Accuracy: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
    print(f"Dev EER:      {mean_eer*100:.3f}% +/- {std_eer*100:.3f}%")

    # Save summary json
    summary_stats = {
        "mean_dev_accuracy": float(mean_acc),
        "std_dev_accuracy": float(std_acc),
        "mean_dev_eer": float(mean_eer),
        "std_dev_eer": float(std_eer),
        "seeds_tested": SEEDS,
    }
    pd.Series(summary_stats).to_json(RESULTS_DIR / "t5_seed_variance_summary.json")

    return res_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_t5(quick=args.quick, full=args.full)
