import argparse
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def compute_eer(y_true, y_scores):
    """
    Computes Equal Error Rate (EER) and corresponding decision threshold.
    EER is where False Acceptance Rate (FAR) == False Rejection Rate (FRR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[idx]
    threshold = thresholds[idx]
    return float(eer), float(threshold)


def load_dataset(pt_path: Path):
    if not pt_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {pt_path}")
    data = torch.load(pt_path)
    X = data["embeddings"].numpy()
    y = data["labels"].numpy()
    filenames = data.get("filenames", [])
    return X, y, filenames


def main():
    parser = argparse.ArgumentParser(description="Train deepfake voice classifier on extracted embeddings")
    parser.add_argument("--embeddings_dir", type=str, default="data/embeddings", help="Path to embeddings folder")
    parser.add_argument("--model_type", type=str, default="logistic_regression",
                        choices=["logistic_regression", "random_forest", "xgboost"],
                        help="Model algorithm to train")
    parser.add_argument("--output_model", type=str, default="models/classifier.joblib", help="Output path for saved model")
    parser.add_argument("--use_mlflow", action="store_true", help="Enable MLflow experiment tracking")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)
    train_path = emb_dir / "train_embeddings.pt"
    dev_path = emb_dir / "dev_embeddings.pt"
    eval_path = emb_dir / "eval_embeddings.pt"

    print("Loading datasets...")
    X_train, y_train, _ = load_dataset(train_path)
    print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} dim")

    has_dev = dev_path.exists()
    has_eval = eval_path.exists()

    if has_dev:
        X_dev, y_dev, _ = load_dataset(dev_path)
        print(f"Dev set: {X_dev.shape[0]} samples")
    if has_eval:
        X_eval, y_eval, _ = load_dataset(eval_path)
        print(f"Eval set: {X_eval.shape[0]} samples")

    print(f"Training {args.model_type} model...")
    if args.model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    elif args.model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    elif args.model_type == "xgboost":
        if not HAS_XGBOOST:
            raise RuntimeError("XGBoost is not installed in the current Python environment.")
        model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)

    if args.use_mlflow and HAS_MLFLOW:
        mlflow.set_experiment("Deepfake_Voice_Detection")
        mlflow.start_run(run_name=f"{args.model_type}_run")
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("train_samples", len(X_train))

    model.fit(X_train, y_train)
    print("Model training completed.")

    # Evaluate on available sets
    eval_sets = [("Train", X_train, y_train)]
    if has_dev:
        eval_sets.append(("Dev", X_dev, y_dev))
    if has_eval:
        eval_sets.append(("Eval", X_eval, y_eval))

    for name, X_set, y_set in eval_sets:
        y_pred = model.predict(X_set)
        y_prob = model.predict_proba(X_set)[:, 1]

        acc = accuracy_score(y_set, y_pred)
        auc = roc_auc_score(y_set, y_prob)
        eer, threshold = compute_eer(y_set, y_prob)

        print(f"\n--- {name} Set Results ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC:  {auc:.4f}")
        print(f"EER:      {eer:.4f} (Threshold: {threshold:.4f})")
        print("Classification Report:")
        print(classification_report(y_set, y_pred, target_names=["bonafide", "spoof"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_set, y_pred))

        if args.use_mlflow and HAS_MLFLOW:
            mlflow.log_metric(f"{name.lower()}_accuracy", acc)
            mlflow.log_metric(f"{name.lower()}_auc", auc)
            mlflow.log_metric(f"{name.lower()}_eer", eer)

    out_file = Path(args.output_model)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_file)
    print(f"\nSaved trained model to {out_file}")

    if args.use_mlflow and HAS_MLFLOW:
        mlflow.end_run()


if __name__ == "__main__":
    main()
