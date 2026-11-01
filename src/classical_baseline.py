"""
Classical Baseline Models
=========================
Provides classical ML classifiers for comparison with our Quantum models.
Uses scikit-learn with cross-validated evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


CLASSICAL_MODELS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
}


def evaluate_model(model, X: np.ndarray, y: np.ndarray,
                   cv: int = 5, random_state: int = 42) -> dict:
    """
    Evaluate a sklearn-compatible model using stratified k-fold CV.
    Returns dict with mean ± std for accuracy, F1, AUC.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    scores = cross_validate(
        model, X, y,
        cv=skf,
        scoring=["accuracy", "f1_weighted", "roc_auc"],
        return_train_score=False,
        n_jobs=-1
    )

    return {
        "accuracy_mean":  float(np.mean(scores["test_accuracy"])),
        "accuracy_std":   float(np.std(scores["test_accuracy"])),
        "f1_mean":        float(np.mean(scores["test_f1_weighted"])),
        "f1_std":         float(np.std(scores["test_f1_weighted"])),
        "auc_mean":       float(np.mean(scores["test_roc_auc"])),
        "auc_std":        float(np.std(scores["test_roc_auc"])),
    }


def run_classical_baseline(X: np.ndarray, y: np.ndarray, cv: int = 5) -> pd.DataFrame:
    """
    Run all classical baseline models and return a results DataFrame.
    """
    results = []
    print("\n" + "="*60)
    print("   CLASSICAL BASELINE EVALUATION")
    print("="*60)

    for name, model in CLASSICAL_MODELS.items():
        print(f"\n[->] Evaluating: {name}...")
        metrics = evaluate_model(model, X, y, cv=cv)
        metrics["model"] = name
        results.append(metrics)
        print(f"    Accuracy: {metrics['accuracy_mean']:.4f} +/- {metrics['accuracy_std']:.4f}")
        print(f"    F1:       {metrics['f1_mean']:.4f} +/- {metrics['f1_std']:.4f}")
        print(f"    AUC:      {metrics['auc_mean']:.4f} +/- {metrics['auc_std']:.4f}")

    df_results = pd.DataFrame(results).set_index("model")
    return df_results


def get_feature_importance(X: np.ndarray, y: np.ndarray,
                            feature_names: list) -> pd.DataFrame:
    """
    Train Random Forest and return feature importance ranking.
    Useful for quantum circuit design — top features → qubits.
    """
    from sklearn.preprocessing import StandardScaler as SS

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


if __name__ == "__main__":
    # Quick sanity test with synthetic data
    from sklearn.datasets import make_classification
    X_test, y_test = make_classification(n_samples=100, n_features=9,
                                          n_informative=5, random_state=42)
    df_res = run_classical_baseline(X_test, y_test, cv=3)
    print("\n\nResults Summary:")
    print(df_res[["accuracy_mean", "f1_mean", "auc_mean"]].to_string())
