"""
Quantum Support Vector Machine (QSVM)
======================================
Uses a quantum kernel (fidelity-based inner product in quantum Hilbert space)
as the kernel function for a classical SVM.

Key insight (Havlíček et al., 2019 — Nature):
  If the quantum feature map φ(x) creates a distribution hard to simulate classically,
  the kernel K(x1,x2) = |⟨φ(x1)|φ(x2)⟩|² may provide a genuine quantum advantage.

Pipeline:
  Raw Features → Scaler → Top-K PCA → Quantum Kernel Matrix → SVM
"""

import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


class QuantumKernelSVM:
    """
    Quantum Support Vector Machine using quantum kernel via fidelity estimation.

    The quantum kernel replaces the classical RBF/polynomial kernel in SVM,
    mapping data to a quantum Hilbert space where classification may be easier.
    """

    def __init__(self,
                 n_qubits: int = 4,
                 encoding: str = "iqp",
                 iqp_reps: int = 2,
                 svm_C: float = 1.0,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Args:
            n_qubits:    Number of qubits (= feature dimension after PCA)
            encoding:    'iqp' (ZZ Feature Map) | 'angle' (RY encoding)
            iqp_reps:    Number of IQP circuit repetitions
            svm_C:       SVM regularization parameter
            random_state: RNG seed
            verbose:     Print progress
        """
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.iqp_reps = iqp_reps
        self.svm_C = svm_C
        self.random_state = random_state
        self.verbose = verbose

        # Classical pre/post processing
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler(feature_range=(0, np.pi))
        self.pca = PCA(n_components=n_qubits, random_state=random_state)

        # SVM with precomputed quantum kernel
        self.svm = SVC(kernel="precomputed", C=svm_C,
                       probability=True, random_state=random_state)

        # Storage
        self.X_train_proc_ = None
        self.K_train_ = None
        self.is_fitted = False

        self._setup_kernel_circuit()

    def _setup_kernel_circuit(self):
        """Build the quantum kernel QNode."""
        try:
            self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
        except Exception:
            self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="numpy")
        def kernel_circuit(x1, x2):
            # Encode x1
            self._encode(x1)
            # Adjoint encode x2 (inverse circuit)
            qml.adjoint(self._encode)(x2)
            # Measure probability of returning to |0⟩ state
            return qml.probs(wires=range(self.n_qubits))

        self._kernel_circuit = kernel_circuit

    def _encode(self, x):
        """Apply feature encoding circuit."""
        if self.encoding == "iqp":
            self._iqp_encode(x)
        else:
            self._angle_encode(x)

    def _angle_encode(self, x):
        """Angle encoding: RY(x_i) on qubit i."""
        for i in range(self.n_qubits):
            qml.RY(float(x[i]), wires=i)

    def _iqp_encode(self, x):
        """IQP ZZ Feature Map encoding (Havlíček et al., 2019)."""
        for _ in range(self.iqp_reps):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for i in range(self.n_qubits):
                qml.PhaseShift(2.0 * float(x[i]), wires=i)
            for i in range(self.n_qubits - 1):
                for j in range(i + 1, self.n_qubits):
                    qml.CNOT(wires=[i, j])
                    angle = 2.0 * (np.pi - float(x[i])) * (np.pi - float(x[j]))
                    qml.PhaseShift(angle, wires=j)
                    qml.CNOT(wires=[i, j])

    def _kernel_value(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute K(x1, x2) = |⟨0|U†(x2)U(x1)|0⟩|²."""
        probs = self._kernel_circuit(x1, x2)
        return float(probs[0])

    def _compute_kernel_matrix(self, X1: np.ndarray,
                                X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix K[i,j] = K(X1[i], X2[j])."""
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        total = n1 * n2
        count = 0

        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_value(X1[i], X2[j])
                count += 1
                if self.verbose and count % max(1, total // 10) == 0:
                    pct = 100 * count / total
                    print(f"  Kernel: {count}/{total} ({pct:.0f}%)")

        return K

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale → PCA → normalize to [0, π] for quantum encoding."""
        if fit:
            X_s = self.scaler.fit_transform(X)
            X_p = self.pca.fit_transform(X_s)
            X_n = self.normalizer.fit_transform(X_p)
        else:
            X_s = self.scaler.transform(X)
            X_p = self.pca.transform(X_s)
            X_n = self.normalizer.transform(X_p)
        return X_n.astype(float)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train QSVM:
          1. Preprocess features
          2. Compute NxN quantum kernel matrix
          3. Fit SVM with precomputed kernel
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Training Quantum SVM (QSVM)")
            print(f"  Qubits: {self.n_qubits} | Encoding: {self.encoding.upper()}")
            print(f"  IQP reps: {self.iqp_reps} | SVM C: {self.svm_C}")
            print(f"  Device: {self.dev.name}")
            print(f"{'='*60}")

        # Preprocess
        X_proc = self._preprocess(X, fit=True)
        self.X_train_proc_ = X_proc

        # Compute training kernel matrix
        if self.verbose:
            print(f"\n[->] Computing {len(X)}x{len(X)} training kernel matrix...")

        self.K_train_ = self._compute_kernel_matrix(X_proc, X_proc)

        if self.verbose:
            print(f"[->] Fitting SVM...")

        self.svm.fit(self.K_train_, y)
        self.is_fitted = True

        if self.verbose:
            train_pred = self.svm.predict(self.K_train_)
            train_acc = accuracy_score(y, train_pred)
            print(f"[OK] QSVM fitted. Train accuracy: {train_acc:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        X_proc = self._preprocess(X, fit=False)
        K_test = self._compute_kernel_matrix(X_proc, self.X_train_proc_)
        return self.svm.predict(K_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        X_proc = self._preprocess(X, fit=False)
        K_test = self._compute_kernel_matrix(X_proc, self.X_train_proc_)
        return self.svm.predict_proba(K_test)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Comprehensive evaluation on test set."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        return {
            "accuracy":         accuracy_score(y, y_pred),
            "f1":               f1_score(y, y_pred, average="weighted", zero_division=0),
            "auc":              roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "report":           classification_report(y, y_pred,
                                                      target_names=["Non-Dementia", "Dementia"])
        }

    def kernel_alignment(self, K: np.ndarray, y: np.ndarray) -> float:
        """
        Compute kernel alignment score — measures how well the quantum kernel
        aligns with the ideal label kernel K_y = yy^T.
        Higher = better quantum-classical alignment.

        Reference: Cristianini et al., 2001
        """
        K_y = np.outer(y * 2 - 1, y * 2 - 1)  # ±1 labels
        alignment = np.trace(K @ K_y) / (
            np.linalg.norm(K, "fro") * np.linalg.norm(K_y, "fro") + 1e-10
        )
        return float(alignment)
