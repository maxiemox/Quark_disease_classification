"""
Jupyter-compatible Demo Notebook Script
=========================================
Run this file cell-by-cell or as a whole to generate the
interactive demo for the hackathon presentation.

Equivalent to running: python main.py --quick
"""

# ╔══════════════════════════════════════════════════╗
# ║  Cell 1: Imports & Setup                         ║
# ╚══════════════════════════════════════════════════╝

import os, sys, warnings
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_parser      import load_oasis_dataset, preprocess
from classical_baseline import run_classical_baseline, get_feature_importance
from hybrid_vqc       import HybridVQC
from quantum_xai      import QuantumXAI, print_explanation
from visualization    import (plot_data_overview, plot_training_loss,
                              plot_confusion_matrix, plot_roc_curves,
                              plot_model_comparison, plot_feature_importance,
                              plot_quantum_advantage_radar)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("✅ All imports successful")
print(f"   PennyLane version: {qml.__version__}")

DISC_ROOT  = r"e:\New folder\Downloads\QuantumX\oasis_cross-sectional_disc1"
OUTPUT_DIR = r"e:\New folder\Downloads\QuantumX\results"
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
FEATURE_NAMES = ["AGE", "SEX_ENC", "HAND_ENC", "EDUC", "SES", "MMSE", "eTIV", "ASF", "nWBV"]
RANDOM_SEED   = 42

# ╔══════════════════════════════════════════════════╗
# ║  Cell 2: Load & Explore Data                     ║
# ╚══════════════════════════════════════════════════╝

df = load_oasis_dataset(DISC_ROOT)
X, y = preprocess(df, binary=True)

print("\n📊 Dataset Info:")
print(f"   Subjects       : {len(df)}")
print(f"   Features       : {list(X.columns)}")
print(f"   Class balance  : {dict(y.value_counts().sort_index())}")
print(f"\n📋 Sample data:\n{df[['AGE','SEX','CDR','MMSE','nWBV']].head(8).to_string()}")

plot_data_overview(df, save_path=os.path.join(OUTPUT_DIR, "plots", "01_dataset_overview.png"))
print("✅ Dataset overview plot saved")

# ╔══════════════════════════════════════════════════╗
# ║  Cell 3: Feature Importance                      ║
# ╚══════════════════════════════════════════════════╝

X_arr, y_arr = X.values, y.values
importance_df = get_feature_importance(X_arr, y_arr, FEATURE_NAMES)
print("\n🔍 Feature Importance Ranking:")
print(importance_df.to_string(index=False))

plot_feature_importance(importance_df,
    save_path=os.path.join(OUTPUT_DIR, "plots", "02_feature_importance.png"))

N_QUBITS = 4
top_features = list(importance_df.head(N_QUBITS)["feature"])
print(f"\n⚛  Top {N_QUBITS} features → Quantum Circuit: {top_features}")

# ╔══════════════════════════════════════════════════╗
# ║  Cell 4: Classical Baseline                      ║
# ╚══════════════════════════════════════════════════╝

print("\n" + "="*55)
print("   Classical Baselines (5-fold CV)")
print("="*55)
classical_df = run_classical_baseline(X_arr, y_arr, cv=5)
print("\n📊 Summary:")
print(classical_df[["accuracy_mean", "f1_mean", "auc_mean"]].to_string())

# ╔══════════════════════════════════════════════════╗
# ║  Cell 5: Quantum Circuit Design                  ║
# ╚══════════════════════════════════════════════════╝

print("\n" + "="*55)
print("   Quantum Circuit (VQC)")
print("="*55)

dev = qml.device("default.qubit", wires=N_QUBITS)
param_shape = qml.StronglyEntanglingLayers.shape(n_layers=3, n_wires=N_QUBITS)

@qml.qnode(dev)
def demo_circuit(x, params):
    # Angle Encoding
    for i in range(N_QUBITS):
        qml.RY(np.pi * x[i], wires=i)
    # Variational Ansatz
    qml.StronglyEntanglingLayers(params, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

x_demo = np.random.uniform(0, 1, N_QUBITS)
p_demo = np.random.uniform(-np.pi, np.pi, param_shape)
print("\n🔬 Circuit Diagram (4 qubits, 3 layers):")
print(qml.draw(demo_circuit)(x_demo, p_demo))
print(f"\n   Total trainable parameters: {np.prod(param_shape)}")

# ╔══════════════════════════════════════════════════╗
# ║  Cell 6: Train Hybrid VQC                        ║
# ╚══════════════════════════════════════════════════╝

X_train, X_test, y_train, y_test = train_test_split(
    X_arr, y_arr, test_size=0.2, random_state=RANDOM_SEED, stratify=y_arr
)
print(f"\n📦 Split: Train={len(X_train)}, Test={len(X_test)}")

vqc = HybridVQC(
    n_qubits=N_QUBITS, n_layers=3, ansatz="strongly_entangling",
    lr=0.01, max_iter=50, random_state=RANDOM_SEED, verbose=True
)
vqc.fit(X_train, y_train, X_val=X_test, y_val=y_test)

vqc_metrics = vqc.evaluate(X_test, y_test)
print(f"\n✅ VQC Results:")
print(f"   Accuracy : {vqc_metrics['accuracy']:.4f}")
print(f"   F1       : {vqc_metrics['f1']:.4f}")
print(f"   AUC-ROC  : {vqc_metrics['auc']:.4f}")

plot_training_loss(vqc.loss_history, vqc.val_acc_history,
    title="Hybrid VQC Training",
    save_path=os.path.join(OUTPUT_DIR, "plots", "03_vqc_training.png"))

plot_confusion_matrix(
    np.array(vqc_metrics["confusion_matrix"]),
    labels=["Non-Dementia", "Dementia"],
    title="Hybrid VQC — Confusion Matrix",
    save_path=os.path.join(OUTPUT_DIR, "plots", "04_vqc_confusion.png"))

# ╔══════════════════════════════════════════════════╗
# ║  Cell 7: Quantum Explainability (XAI)            ║
# ╚══════════════════════════════════════════════════╝

print("\n" + "="*55)
print("   🔍 Quantum Explainability Analysis (QuantumXAI)")
print("="*55)

xai = QuantumXAI(vqc, n_qubits=N_QUBITS, n_layers=3)
params = np.array(vqc.params_)
X_test_proc = vqc._preprocess(X_test, fit=False)
x_sample = X_test_proc[0]

# Feature attribution
print("\n[1] Quantum Feature Attributions (SHAP-inspired):")
attributions = xai.feature_attribution(x_sample, params, n_perturbations=10)
for fname, attr in sorted(zip(top_features, attributions), key=lambda t: t[1], reverse=True):
    bar = "█" * int(attr * 30)
    print(f"  {fname:<12}: {bar:<30} {attr:.3f}")

# Entanglement entropy
print("\n[2] Entanglement Entropy:")
ent = xai.entanglement_entropy(x_sample, params)
print(f"  Von Neumann Entropy  : {ent['von_neumann_entropy']:.4f} bits")
print(f"  Normalized Entropy   : {ent['normalized_entropy']:.4f}")
print(f"  Quantum Entanglement : {'YES ✅' if ent['is_entangled'] else 'NO ⚠️'}")

# Barren plateau
print("\n[3] Barren Plateau Analysis:")
bp = xai.barren_plateau_analysis(x_sample, n_random_inits=15, param_shape=vqc.param_shape)
print(f"  Mean Gradient Variance : {bp['mean_gradient_variance']:.6f}")
print(f"  Barren Plateau         : {'YES ⚠️' if bp['barren_plateau_detected'] else 'NO ✅'}")

# Full prediction explanation
print("\n[4] Per-Patient Quantum Explanation:")
explanation = xai.explain_prediction(
    X_test[0], top_features, params,
    preprocessor=lambda x: vqc._preprocess(x.reshape(1, -1), fit=False)[0]
)
print_explanation(explanation)

# ╔══════════════════════════════════════════════════╗
# ║  Cell 8: Final Comparison                        ║
# ╚══════════════════════════════════════════════════╝

print("\n" + "="*55)
print("   📊 Final Model Comparison")
print("="*55)

all_results = classical_df.to_dict(orient="index")
all_results["Hybrid VQC"] = {
    "accuracy_mean": vqc_metrics["accuracy"], "accuracy_std": 0.0,
    "f1_mean": vqc_metrics["f1"],             "f1_std": 0.0,
    "auc_mean": vqc_metrics["auc"],            "auc_std": 0.0,
}
combined = pd.DataFrame(all_results).T

print(f"\n{'Model':<25} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
print("-"*55)
for mname, row in combined.iterrows():
    tag = " ⚛" if any(k in mname for k in ["Quantum", "VQC", "Hybrid"]) else ""
    print(f"{mname+tag:<27} {row['accuracy_mean']:>10.4f} {row['f1_mean']:>10.4f} {row['auc_mean']:>10.4f}")

plot_model_comparison(combined,
    save_path=os.path.join(OUTPUT_DIR, "plots", "07_model_comparison.png"))
plot_quantum_advantage_radar(combined,
    save_path=os.path.join(OUTPUT_DIR, "plots", "08_radar.png"))

# ROC Curves
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

roc_data = {"Hybrid VQC": (y_test, vqc.predict_proba(X_test)[:, 1])}
for cname, cpipe in [
    ("SVM (Classical)", Pipeline([("s", StandardScaler()), ("c", SVC(probability=True, random_state=42))])),
    ("Logistic Regression", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000, random_state=42))])),
]:
    cpipe.fit(X_train, y_train)
    roc_data[cname] = (y_test, cpipe.predict_proba(X_test)[:, 1])

plot_roc_curves(roc_data,
    save_path=os.path.join(OUTPUT_DIR, "plots", "09_roc_curves.png"))

print("\n✅ All plots generated!")
print(f"📁 Results: {OUTPUT_DIR}/plots/")
print("\n🏆 Demo complete. Good luck at the hackathon! ⚛️🧠")
