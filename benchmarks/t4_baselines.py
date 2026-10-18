import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

from benchmarks.utils import (
    compute_eer,
    get_classifier,
    RANDOM_SEED,
    CACHE_DIR,
    RESULTS_DIR,
)

def run_t4(quick: bool = False, full: bool = False):
    print("=" * 60)
    print("Running T4: Baselines Comparison on Cached Embeddings")
    print("=" * 60)

    # Load Train & Dev embeddings
    train_data = torch.load("data/embeddings/train_embeddings.pt", weights_only=False)
    dev_data = torch.load("data/embeddings/dev_embeddings.pt", weights_only=False)

    X_train = train_data["embeddings"].float().numpy()
    y_train = train_data["labels"].numpy()

    X_dev = dev_data["embeddings"].float().numpy()
    y_dev = dev_data["labels"].numpy()

    if quick:
        # Subsample for very quick verification
        idx_tr = np.random.RandomState(RANDOM_SEED).choice(len(X_train), size=2500, replace=False)
        X_train, y_train = X_train[idx_tr], y_train[idx_tr]
        idx_dev = np.random.RandomState(RANDOM_SEED).choice(len(X_dev), size=2500, replace=False)
        X_dev, y_dev = X_dev[idx_dev], y_dev[idx_dev]

    # Load eval embeddings if available
    eval_cache_name = "eval_embeddings_full.pt" if full else ("eval_embeddings_quick.pt" if quick else "eval_embeddings_stratified_3000.pt")
    eval_cache = CACHE_DIR / eval_cache_name
    X_eval, y_eval = None, None
    if eval_cache.exists():
        eval_data = torch.load(eval_cache, weights_only=False)
        X_eval = eval_data["embeddings"].float().numpy()
        y_eval = eval_data["labels"].numpy()

    models = {
        "Majority Class (Dummy)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"),
        "Linear SVM": LinearSVC(max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced"),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    results = []

    # 1. Evaluate baseline models
    for name, clf in models.items():
        print(f"Fitting {name}...")
        clf.fit(X_train, y_train)

        # Predict dev
        if hasattr(clf, "predict_proba"):
            dev_scores = clf.predict_proba(X_dev)[:, 1]
        elif hasattr(clf, "decision_function"):
            raw_scores = clf.decision_function(X_dev)
            # Sigmoid scaling for decision function
            dev_scores = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            dev_scores = np.zeros(len(X_dev))

        dev_eer, dev_th, _, _, _ = compute_eer(y_dev, dev_scores)
        try:
            dev_auc = roc_auc_score(y_dev, dev_scores)
        except Exception:
            dev_auc = float("nan")

        row = {
            "Model": name,
            "Dev_EER": float(dev_eer),
            "Dev_AUC": float(dev_auc),
            "Dev_EER_Threshold": float(dev_th),
        }

        # Predict eval if available
        if X_eval is not None and y_eval is not None:
            if hasattr(clf, "predict_proba"):
                eval_scores = clf.predict_proba(X_eval)[:, 1]
            elif hasattr(clf, "decision_function"):
                raw_scores = clf.decision_function(X_eval)
                eval_scores = 1.0 / (1.0 + np.exp(-raw_scores))
            else:
                eval_scores = np.zeros(len(X_eval))

            eval_eer, eval_th, _, _, _ = compute_eer(y_eval, eval_scores)
            try:
                eval_auc = roc_auc_score(y_eval, eval_scores)
            except Exception:
                eval_auc = float("nan")

            row["Eval_EER"] = float(eval_eer)
            row["Eval_AUC"] = float(eval_auc)
            row["Eval_EER_Threshold"] = float(eval_th)

        results.append(row)

    # 2. Add Frozen PyTorch MLP Classifier
    pt_clf = get_classifier()
    with torch.no_grad():
        mlp_dev_probs = torch.softmax(pt_clf(torch.from_numpy(X_dev).to(pt_clf.net[0].weight.device)), dim=-1)[:, 1].cpu().numpy()
    mlp_dev_eer, mlp_dev_th, _, _, _ = compute_eer(y_dev, mlp_dev_probs)
    mlp_dev_auc = roc_auc_score(y_dev, mlp_dev_probs)

    mlp_row = {
        "Model": "Deepfake MLP (Wav2Vec2 + 768-256-64-2)",
        "Dev_EER": float(mlp_dev_eer),
        "Dev_AUC": float(mlp_dev_auc),
        "Dev_EER_Threshold": float(mlp_dev_th),
    }

    if X_eval is not None and y_eval is not None:
        with torch.no_grad():
            mlp_eval_probs = torch.softmax(pt_clf(torch.from_numpy(X_eval).to(pt_clf.net[0].weight.device)), dim=-1)[:, 1].cpu().numpy()
        mlp_eval_eer, mlp_eval_th, _, _, _ = compute_eer(y_eval, mlp_eval_probs)
        mlp_eval_auc = roc_auc_score(y_eval, mlp_eval_probs)
        mlp_row["Eval_EER"] = float(mlp_eval_eer)
        mlp_row["Eval_AUC"] = float(mlp_eval_auc)
        mlp_row["Eval_EER_Threshold"] = float(mlp_eval_th)

    results.append(mlp_row)

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / "t4_baselines.csv", index=False)
    print(f"Baselines comparison saved to {RESULTS_DIR / 't4_baselines.csv'}")

    print("\n--- Model Baseline Comparison ---")
    cols = ["Model", "Dev_EER", "Dev_AUC"]
    if "Eval_EER" in res_df.columns:
        cols += ["Eval_EER", "Eval_AUC"]
    print(res_df[cols].to_string(index=False))

    return res_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_t4(quick=args.quick, full=args.full)
