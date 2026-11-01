"""
Hybrid Quantum-Classical Classifier
=====================================
Combines a classical feature extractor (StandardScaler + PCA-style reduction)
with a Variational Quantum Classifier (VQC) trained via parameter-shift gradients.

Architecture:
  Raw Features (9D)
       ↓
  StandardScaler
       ↓
  PCA → n_qubits dimensions
       ↓
  [QUANTUM LAYER] VQC with StronglyEntanglingLayers
       ↓
  ⟨Z₀⟩ expectation → sigmoid → binary prediction

Training uses Adam optimizer via PennyLane's autograd interface.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


class HybridVQC:
    """
    Hybrid Quantum-Classical Variational Quantum Classifier.

    The 'hybrid' nature:
      - Classical preprocessing (StandardScaler + PCA) reduces dimensionality
        to match qubit count, enabling efficient quantum encoding.
      - Quantum variational circuit learns optimal measurement basis.
      - Classical sigmoid post-processes quantum expectation values.

    This reflects real NISQ (Noisy Intermediate-Scale Quantum) deployment
    where limited qubits are augmented by classical layers.
    """

    def __init__(self,
                 n_qubits: int = 4,
                 n_layers: int = 3,
                 ansatz: str = "strongly_entangling",
                 lr: float = 0.01,
                 max_iter: int = 100,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Args:
            n_qubits:     Number of qubits (also = reduced feature dimension)
            n_layers:     Depth of variational ansatz
            ansatz:       'strongly_entangling' | 'hardware_efficient'
            lr:           Learning rate for Adam optimizer
            max_iter:     Number of training epochs
            random_state: RNG seed
            verbose:      Print training progress
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        self.lr = lr
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        np.random.seed(random_state)

        # Classical preprocessing
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler(feature_range=(0, 1))
        self.pca = PCA(n_components=n_qubits, random_state=random_state)

        # Quantum device
        self._setup_quantum_circuit()

        # Training history
        self.loss_history = []
        self.val_acc_history = []
        self.params_ = None
        self.is_fitted = False

    def _setup_quantum_circuit(self):
        """Initialize PennyLane device and build the QNode."""
        try:
            self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
        except Exception:
            self.dev = qml.device("default.qubit", wires=self.n_qubits)

        if self.ansatz == "strongly_entangling":
            self.param_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=self.n_layers, n_wires=self.n_qubits
            )
        else:
            self.param_shape = (self.n_layers, self.n_qubits, 2)

        @qml.qnode(self.dev, interface="autograd", diff_method="parameter-shift")
        def _circuit(x, params):
            # ── Encoding ──────────────────────────────────────────────────
            # Angle encoding: feature x_i → RY(π·x_i) gate on qubit i
            for i in range(self.n_qubits):
                qml.RY(pnp.pi * x[i], wires=i)

            # ── Ansatz ────────────────────────────────────────────────────
            if self.ansatz == "strongly_entangling":
                qml.StronglyEntanglingLayers(params, wires=range(self.n_qubits))
            else:
                for layer in range(self.n_layers):
                    for i in range(self.n_qubits):
                        qml.RY(params[layer, i, 0], wires=i)
                        qml.RZ(params[layer, i, 1], wires=i)
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if self.n_qubits > 2:
                        qml.CNOT(wires=[self.n_qubits - 1, 0])

            # ── Measurement ───────────────────────────────────────────────
            return qml.expval(qml.PauliZ(0))

        self._circuit = _circuit

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale → PCA → normalize to [0,1] for angle encoding."""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_scaled)
            X_norm = self.normalizer.fit_transform(X_pca)
        else:
            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled)
            X_norm = self.normalizer.transform(X_pca)
        return X_norm.astype(float)

    def _predict_single(self, x: np.ndarray, params):
        """Get probability (via sigmoid on <Z>) for a single sample.
        Returns autograd-compatible value during training, float during inference."""
        exp_val = self._circuit(x, params)
        return (1 + exp_val) / 2  # Map [-1,1] -> [0,1]

    def _binary_cross_entropy(self, params, X_batch: np.ndarray,
                               y_batch: np.ndarray):
        """Binary cross-entropy loss over a mini-batch (autograd-compatible)."""
        eps = 1e-8
        loss = 0.0
        for x, label in zip(X_batch, y_batch):
            p = self._predict_single(x, params)
            # Use pnp.clip to stay within autograd graph
            p = pnp.clip(p, eps, 1 - eps)
            loss = loss - (label * pnp.log(p) + (1 - label) * pnp.log(1 - p))
        return loss / len(X_batch)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train the VQC using Adam optimizer with parameter-shift gradients.

        Args:
            X, y:         Training data
            X_val, y_val: Optional validation set for monitoring
        """
        # ── Preprocess ────────────────────────────────────────────────────
        X_proc = self._preprocess(X, fit=True)
        if X_val is not None:
            X_val_proc = self._preprocess(X_val, fit=False)

        # ── Initialize parameters ─────────────────────────────────────────
        rng = np.random.default_rng(self.random_state)
        params = pnp.array(
            rng.uniform(-np.pi, np.pi, self.param_shape),
            requires_grad=True
        )

        optimizer = qml.AdamOptimizer(stepsize=self.lr)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Training Hybrid VQC")
            print(f"  Qubits: {self.n_qubits} | Layers: {self.n_layers} | "
                  f"Ansatz: {self.ansatz}")
            print(f"  Device: {self.dev.name}")
            print(f"  Params: {np.prod(self.param_shape):,}")
            print(f"{'='*60}")

        # ── Training loop ─────────────────────────────────────────────────
        for epoch in range(self.max_iter):
            # Mini-batch gradient (full batch for small datasets)
            def loss_fn(p):
                return self._binary_cross_entropy(p, X_proc, y)

            params, loss = optimizer.step_and_cost(loss_fn, params)
            self.loss_history.append(float(loss))

            # Validation accuracy
            if X_val is not None and (epoch + 1) % 10 == 0:
                y_pred = self.predict(X_val)
                val_acc = accuracy_score(y_val, y_pred)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print(f"  Epoch {epoch+1:3d}/{self.max_iter} | "
                          f"Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")
            elif self.verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.max_iter} | Loss: {loss:.4f}")

        self.params_ = params
        self.is_fitted = True

        # Calibrate optimal decision threshold on training set
        train_probs = self.predict_proba(X)[:, 1]
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.15, 0.75, 61):
            preds = (train_probs >= t).astype(int)
            f1 = f1_score(y, preds, average="weighted", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        self.best_threshold_ = float(best_t)

        if self.verbose:
            print(f"\n[OK] Training complete. Final loss: {self.loss_history[-1]:.4f} (Calibrated threshold: {self.best_threshold_:.3f})")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of class 1 for each sample."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_proc = self._preprocess(X, fit=False)
        probs = np.array([float(self._predict_single(x, self.params_)) for x in X_proc])
        return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Return binary class predictions using calibrated threshold."""
        proba = self.predict_proba(X)[:, 1]
        t = getattr(self, "best_threshold_", 0.5) if threshold is None else threshold
        return (proba >= t).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on (X, y)."""
        return accuracy_score(y, self.predict(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Return comprehensive evaluation metrics."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1":       f1_score(y, y_pred, average="weighted", zero_division=0),
            "auc":      roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }

    def get_circuit_diagram(self) -> str:
        """Return ASCII circuit diagram."""
        x_sample = np.random.uniform(0, 1, self.n_qubits)
        p_sample = np.random.uniform(-np.pi, np.pi, self.param_shape)
        return qml.draw(self._circuit)(x_sample, p_sample)

    def n_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return int(np.prod(self.param_shape))
