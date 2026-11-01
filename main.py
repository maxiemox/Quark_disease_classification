"""
QuantumX — Main Pipeline Runner
================================
Alzheimer's Disease Classification using Hybrid Quantum-Classical ML
on the OASIS Cross-Sectional Brain MRI Dataset.

Run this script to execute the full pipeline:
  1. Parse & preprocess OASIS dataset
  2. Classical baseline evaluation (CV)
  3. Train Hybrid VQC
  4. Train Quantum SVM (QSVM)
  5. Quantum Explainability Analysis (XAI)
  6. Generate all plots and reports
  7. Save results summary

Usage:
    python main.py
    python main.py --qubits 5 --layers 4 --epochs 80
    python main.py --quick  (fast run for demo)
"""

import os
import sys
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
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data_parser      import load_oasis_dataset, preprocess
from classical_baseline import run_classical_baseline, get_feature_importance
from hybrid_vqc       import HybridVQC
from qsvm             import QuantumKernelSVM
from quantum_xai      import QuantumXAI, print_explanation
from visualization    import (
    plot_training_loss, plot_confusion_matrix, plot_roc_curves,
    plot_model_comparison, plot_feature_importance,
    plot_kernel_matrix, plot_data_overview,
    plot_quantum_advantage_radar,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DISC_ROOT   = r"e:\New folder\Downloads\QuantumX\oasis_cross-sectional_disc1"
OUTPUT_DIR  = r"e:\New folder\Downloads\QuantumX\results"
RANDOM_SEED = 42

FEATURE_NAMES = ["AGE", "SEX_ENC", "HAND_ENC", "EDUC", "SES",
                 "MMSE", "eTIV", "ASF", "nWBV"]


def parse_args():
    parser = argparse.ArgumentParser(description="QuantumX: QML for Alzheimer's Detection")
    parser.add_argument("--qubits",   type=int, default=4,   help="Number of qubits")
    parser.add_argument("--layers",   type=int, default=3,   help="VQC circuit layers")
    parser.add_argument("--epochs",   type=int, default=60,  help="VQC training epochs")
    parser.add_argument("--cv",       type=int, default=5,   help="Cross-validation folds")
    parser.add_argument("--lr",       type=float, default=0.01, help="VQC learning rate")
    parser.add_argument("--quick",    action="store_true",   help="Quick demo mode")
    parser.add_argument("--skip-qsvm", action="store_true",  help="Skip QSVM (slow)")
    parser.add_argument("--skip-xai",  action="store_true",  help="Skip XAI analysis")
    return parser.parse_args()


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
    print(f"[INFO] Output directory: {OUTPUT_DIR}")


def banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║    ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██╗   ██╗║
║   ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██║   ██║║
║   ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██║   ██║║
║   ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║   ██║║
║   ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ╚██████╔╝║
║    ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ║
║                                                          ║
║   Quantum Machine Learning for Alzheimer's Detection     ║
║   OASIS Brain MRI Dataset · PennyLane · SDG-3           ║
╚══════════════════════════════════════════════════════════╝
    """)


def main():
    args = parse_args()
    banner()
    ensure_output_dir()

    if args.quick:
        print("[INFO] Quick mode: reduced epochs & no QSVM kernel computation")
        args.epochs   = 20
        args.layers   = 2
        args.skip_qsvm = True

    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: DATA LOADING & PREPROCESSING
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("  STEP 1: Data Loading & Preprocessing")
    print("─"*60)

    df = load_oasis_dataset(DISC_ROOT)
    X, y = preprocess(df, binary=True)

    # Data overview plot
    print("[→] Generating dataset overview plot...")
    plot_data_overview(df, save_path=os.path.join(plots_dir, "01_dataset_overview.png"))

    X_arr = X.values
    y_arr = y.values

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=RANDOM_SEED, stratify=y_arr
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # Feature importance analysis
    print("[→] Analyzing feature importance...")
    importance_df = get_feature_importance(X_arr, y_arr, FEATURE_NAMES)
    print(importance_df.to_string(index=False))
    plot_feature_importance(importance_df,
                            save_path=os.path.join(plots_dir, "02_feature_importance.png"))

    # Top features → will be used as quantum circuit features
    top_features = list(importance_df.head(args.qubits)["feature"])
    print(f"\n[INFO] Top {args.qubits} features for quantum circuit: {top_features}")
    results["top_quantum_features"] = top_features

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: CLASSICAL BASELINE
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("  STEP 2: Classical Baseline Evaluation")
    print("─"*60)

    classical_results = run_classical_baseline(X_arr, y_arr, cv=args.cv)
    print("\n[Classical Results Summary]")
    print(classical_results[["accuracy_mean", "f1_mean", "auc_mean"]].to_string())
    results["classical"] = classical_results.to_dict()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: HYBRID VARIATIONAL QUANTUM CLASSIFIER (VQC)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("  STEP 3: Hybrid VQC Training")
    print("─"*60)

    vqc = HybridVQC(
        n_qubits=args.qubits,
        n_layers=args.layers,
        ansatz="strongly_entangling",
        lr=args.lr,
        max_iter=args.epochs,
        random_state=RANDOM_SEED,
        verbose=True,
    )

    # Print circuit diagram
    print("\n[Quantum Circuit Diagram]")
    print(vqc.get_circuit_diagram())
    print(f"[INFO] Total trainable parameters: {vqc.n_parameters():,}")

    # Train
    vqc.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    # Evaluate
    vqc_metrics = vqc.evaluate(X_test, y_test)
    print(f"\n[VQC Test Results]")
    print(f"  Accuracy : {vqc_metrics['accuracy']:.4f}")
    print(f"  F1 Score : {vqc_metrics['f1']:.4f}")
    print(f"  AUC-ROC  : {vqc_metrics['auc']:.4f}")

    results["vqc"] = {
        "accuracy_mean": vqc_metrics["accuracy"],
        "f1_mean":       vqc_metrics["f1"],
        "auc_mean":      vqc_metrics["auc"],
        "accuracy_std":  0.0,
        "f1_std":        0.0,
        "auc_std":       0.0,
        "n_params":      vqc.n_parameters(),
        "n_qubits":      args.qubits,
        "n_layers":      args.layers,
        "epochs":        args.epochs,
    }

    # Training curve
    plot_training_loss(
        vqc.loss_history, vqc.val_acc_history,
        title="Hybrid VQC Training (OASIS Alzheimer's Dataset)",
        save_path=os.path.join(plots_dir, "03_vqc_training_curve.png")
    )

    # Confusion matrix
    plot_confusion_matrix(
        np.array(vqc_metrics["confusion_matrix"]),
        labels=["Non-Dementia", "Dementia"],
        title="VQC Confusion Matrix",
        save_path=os.path.join(plots_dir, "04_vqc_confusion_matrix.png")
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: QUANTUM SVM (QSVM)
    # ─────────────────────────────────────────────────────────────────────
    if not args.skip_qsvm:
        print("\n" + "─"*60)
        print("  STEP 4: Quantum SVM Training")
        print("─"*60)

        qsvm = QuantumKernelSVM(
            n_qubits=args.qubits,
            encoding="iqp",
            iqp_reps=2,
            svm_C=1.0,
            random_state=RANDOM_SEED,
            verbose=True,
        )
        qsvm.fit(X_train, y_train)
        qsvm_metrics = qsvm.evaluate(X_test, y_test)

        print(f"\n[QSVM Test Results]")
        print(f"  Accuracy : {qsvm_metrics['accuracy']:.4f}")
        print(f"  F1 Score : {qsvm_metrics['f1']:.4f}")
        print(f"  AUC-ROC  : {qsvm_metrics['auc']:.4f}")
        print(qsvm_metrics["report"])

        results["qsvm"] = {
            "accuracy_mean": qsvm_metrics["accuracy"],
            "f1_mean":       qsvm_metrics["f1"],
            "auc_mean":      qsvm_metrics["auc"],
            "accuracy_std":  0.0,
            "f1_std":        0.0,
            "auc_std":       0.0,
        }

        # Kernel alignment
        X_train_proc = qsvm._preprocess(X_train, fit=False)
        alignment = qsvm.kernel_alignment(qsvm.K_train_, y_train)
        results["qsvm"]["kernel_alignment"] = alignment
        print(f"\n[QSVM] Quantum Kernel Alignment: {alignment:.4f}")

        # Kernel matrix heatmap
        plot_kernel_matrix(
            qsvm.K_train_, y_train,
            title="Quantum Kernel Matrix (IQP ZZ Feature Map)",
            save_path=os.path.join(plots_dir, "05_quantum_kernel_matrix.png")
        )

        # Confusion matrix
        plot_confusion_matrix(
            np.array(qsvm_metrics["confusion_matrix"]),
            labels=["Non-Dementia", "Dementia"],
            title="QSVM Confusion Matrix",
            save_path=os.path.join(plots_dir, "06_qsvm_confusion_matrix.png")
        )
    else:
        print("\n[STEP 4] QSVM skipped (--skip-qsvm or quick mode)")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: QUANTUM EXPLAINABILITY ANALYSIS (XAI)
    # ─────────────────────────────────────────────────────────────────────
    if not args.skip_xai and vqc.is_fitted:
        print("\n" + "─"*60)
        print("  STEP 5: Quantum Explainability Analysis (QuantumXAI)")
        print("─"*60)

        xai = QuantumXAI(vqc, n_qubits=args.qubits, n_layers=args.layers)
        params = np.array(vqc.params_)

        # Process a sample for XAI
        X_test_proc = vqc._preprocess(X_test, fit=False)
        sample_idx = 0
        x_sample = X_test_proc[sample_idx]

        # Feature attribution
        print("\n[→] Computing quantum feature attributions...")
        attributions = xai.feature_attribution(x_sample, params, n_perturbations=15)
        attr_df = pd.DataFrame({
            "feature": FEATURE_NAMES[:args.qubits],
            "quantum_attribution": attributions
        }).sort_values("quantum_attribution", ascending=False)
        print("\nQuantum Feature Attributions:")
        print(attr_df.to_string(index=False))
        results["xai_attributions"] = attr_df.to_dict(orient="records")

        # Barren plateau analysis
        print("\n[→] Barren plateau analysis...")
        bp_result = xai.barren_plateau_analysis(
            x_sample, n_random_inits=20, param_shape=vqc.param_shape
        )
        results["barren_plateau"] = {k: v for k, v in bp_result.items()
                                     if not isinstance(v, np.ndarray)}

        # Entanglement entropy
        print("\n[→] Computing entanglement entropy...")
        ent_result = xai.entanglement_entropy(x_sample, params)
        print(f"  Von Neumann Entropy: {ent_result['von_neumann_entropy']:.4f} bits")
        print(f"  Normalized Entropy:  {ent_result['normalized_entropy']:.4f}")
        print(f"  Entangled:           {ent_result['is_entangled']}")
        results["entanglement"] = ent_result

        # Layer sensitivity
        print("\n[→] Layer sensitivity analysis...")
        layer_sens = xai.layer_sensitivity_summary(X_test_proc[:5], params)
        for i, sens in enumerate(layer_sens):
            print(f"  Layer {i+1}: sensitivity = {sens:.4f}")
        results["layer_sensitivity"] = layer_sens.tolist()

        # Example prediction explanation
        print("\n[→] Generating prediction explanation for test sample...")
        raw_sample = X_test[sample_idx]
        explanation = xai.explain_prediction(
            raw_sample, FEATURE_NAMES[:args.qubits],
            params,
            preprocessor=lambda x: vqc._preprocess(x.reshape(1,-1), fit=False)[0]
        )
        print_explanation(explanation)
        results["example_explanation"] = explanation

    # ─────────────────────────────────────────────────────────────────────
    # STEP 6: COMBINED COMPARISON PLOTS
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("  STEP 6: Generating Comparison Plots")
    print("─"*60)

    # Build combined results dataframe
    all_model_results = {}

    # Add classical
    for model_name, row in classical_results.iterrows():
        all_model_results[model_name] = {
            "accuracy_mean": row["accuracy_mean"],
            "accuracy_std":  row["accuracy_std"],
            "f1_mean":       row["f1_mean"],
            "f1_std":        row["f1_std"],
            "auc_mean":      row["auc_mean"],
            "auc_std":       row["auc_std"],
        }

    # Add quantum
    all_model_results["Hybrid VQC"] = results["vqc"]
    if "qsvm" in results:
        all_model_results["QSVM"] = results["qsvm"]

    combined_df = pd.DataFrame(all_model_results).T

    # Model comparison bar chart
    plot_model_comparison(
        combined_df,
        save_path=os.path.join(plots_dir, "07_model_comparison.png")
    )

    # Radar chart
    plot_quantum_advantage_radar(
        combined_df,
        save_path=os.path.join(plots_dir, "08_radar_chart.png")
    )

    # ROC curves
    print("[→] Computing ROC curves...")
    roc_data = {}
    vqc_proba = vqc.predict_proba(X_test)[:, 1]
    roc_data["Hybrid VQC"] = (y_test, vqc_proba)

    if "qsvm" in results and not args.skip_qsvm:
        qsvm_proba = qsvm.predict_proba(X_test)[:, 1]
        roc_data["QSVM"] = (y_test, qsvm_proba)

    # Classical ROC (train fresh on full split)
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    for clf_name, clf_pipe in [
        ("SVM (Classical)", Pipeline([("s", StandardScaler()), ("c", SVC(probability=True, random_state=RANDOM_SEED))])),
        ("Logistic Regression", Pipeline([("s", StandardScaler()), ("c", __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))])),
    ]:
        clf_pipe.fit(X_train, y_train)
        roc_data[clf_name] = (y_test, clf_pipe.predict_proba(X_test)[:, 1])

    plot_roc_curves(roc_data,
                    save_path=os.path.join(plots_dir, "09_roc_curves.png"))

    # ─────────────────────────────────────────────────────────────────────
    # STEP 7: SAVE RESULTS SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("  STEP 7: Saving Results")
    print("─"*60)

    summary = {
        "timestamp":        timestamp,
        "dataset":          "OASIS Cross-Sectional",
        "n_subjects":       len(df),
        "n_features":       len(FEATURE_NAMES),
        "task":             "Binary: Dementia vs Non-Dementia",
        "vqc_config": {
            "n_qubits": args.qubits,
            "n_layers": args.layers,
            "n_params": results["vqc"]["n_params"],
            "epochs":   args.epochs,
            "lr":       args.lr,
        },
        "model_results":    {
            name: {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in row.items()
                   if isinstance(v, (int, float, np.floating, str))}
            for name, row in combined_df.iterrows()
        },
        "top_quantum_features": top_features,
    }

    if "barren_plateau" in results:
        summary["barren_plateau"] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in results.get("barren_plateau", {}).items()
        }
    if "entanglement" in results:
        summary["entanglement_entropy"] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in results.get("entanglement", {}).items()
        }

    summary_path = os.path.join(OUTPUT_DIR, f"results_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[✓] Results saved to: {summary_path}")

    # ─────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  FINAL RESULTS SUMMARY")
    print("═"*60)
    print(f"\n{'Model':<25} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-"*60)
    for model_name, row in combined_df.iterrows():
        marker = " ⚛" if any(k in model_name for k in ["Quantum", "VQC", "QSVM", "Hybrid"]) else ""
        print(f"{model_name+marker:<27} {row['accuracy_mean']:>10.4f} "
              f"{row['f1_mean']:>10.4f} {row['auc_mean']:>10.4f}")

    print("\n⚛ = Quantum model")
    print("\n[✓] All plots saved to:", os.path.join(OUTPUT_DIR, "plots"))
    print("[✓] Pipeline complete!\n")

    return summary


if __name__ == "__main__":
    main()
