"""
Quantum Explainability Module (QuantumXAI) — UNIQUE INNOVATION
================================================================
This module brings explainability to Quantum ML — a rarely explored frontier.

Key ideas implemented:
  1. Quantum Parameter Sensitivity (QPS): How sensitive is the model output
     to each variational parameter? High sensitivity → high influence.
     Uses parameter-shift rule: ∂⟨O⟩/∂θ_i = [⟨O⟩(θ_i+π/2) - ⟨O⟩(θ_i-π/2)] / 2

  2. Quantum Feature Attribution (QFA): Which INPUT features contribute most
     to the quantum model's prediction on a given sample?
     Implemented via quantum analogue of SHAP (Shapley-inspired perturbation).

  3. Barren Plateau Detector: Checks if gradient variance is exponentially
     small (sign of barren plateau — a fundamental QML training challenge).
     Computed as Var[∂L/∂θ_i] across random parameter initializations.

  4. Entanglement Entropy Monitor: Measures bipartite entanglement of the
     quantum state during inference — a uniquely quantum metric showing
     the circuit is leveraging quantum correlations.

SDG Relevance:
  SDG 3 (Good Health) — Explainable AI is critical for clinical adoption.
  Doctors need to trust and understand why a model flags a patient.
  QuantumXAI bridges the quantum black-box gap.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import warnings
warnings.filterwarnings("ignore")


class QuantumXAI:
    """
    Quantum Explainability and Interpretability Tool for the Hybrid VQC.

    Provides unique insights into quantum model behavior that have no
    classical equivalent — our key differentiator for the hackathon.
    """

    def __init__(self, vqc_model, n_qubits: int = 4, n_layers: int = 3):
        """
        Args:
            vqc_model: A fitted HybridVQC instance
            n_qubits:  Number of qubits
            n_layers:  Number of variational layers
        """
        self.model = vqc_model
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Separate device for XAI computations (avoid side-effects)
        try:
            self.dev = qml.device("lightning.qubit", wires=n_qubits)
        except Exception:
            self.dev = qml.device("default.qubit", wires=n_qubits)

    # ─────────────────────────────────────────────────────────────────────
    # 1. QUANTUM PARAMETER SENSITIVITY (QPS)
    # ─────────────────────────────────────────────────────────────────────

    def parameter_sensitivity(self, X_sample: np.ndarray,
                               params: np.ndarray) -> np.ndarray:
        """
        Compute gradient of output w.r.t each variational parameter
        using the parameter-shift rule (exact quantum gradient).

        ∂⟨Z⟩/∂θ_k = [⟨Z⟩(θ_k + π/2) - ⟨Z⟩(θ_k - π/2)] / 2

        Returns:
            gradients: Array of same shape as params
                       (n_layers × n_qubits × n_angles)
        """
        gradients = np.zeros_like(params)
        flat_params = params.flatten()
        n_params = len(flat_params)

        @qml.qnode(self.dev, interface="numpy")
        def circuit(x, p_flat):
            p = p_flat.reshape(params.shape)
            for i in range(self.n_qubits):
                qml.RY(np.pi * float(x[i]), wires=i)
            qml.StronglyEntanglingLayers(p, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))

        for k in range(n_params):
            p_plus = flat_params.copy()
            p_minus = flat_params.copy()
            p_plus[k]  += np.pi / 2
            p_minus[k] -= np.pi / 2
            grad_k = (circuit(X_sample, p_plus) - circuit(X_sample, p_minus)) / 2.0
            gradients.flat[k] = grad_k

        return gradients

    def layer_sensitivity_summary(self, X_samples: np.ndarray,
                                   params: np.ndarray) -> np.ndarray:
        """
        Average |gradient| per layer across a set of samples.
        Returns array of shape (n_layers,) — higher = more influential layer.
        """
        layer_grads = np.zeros(self.n_layers)

        for x in X_samples:
            grads = self.parameter_sensitivity(x, params)
            layer_grads += np.mean(np.abs(grads), axis=(1, 2))

        return layer_grads / len(X_samples)

    # ─────────────────────────────────────────────────────────────────────
    # 2. QUANTUM FEATURE ATTRIBUTION (QFA)
    # ─────────────────────────────────────────────────────────────────────

    def feature_attribution(self, X_sample: np.ndarray,
                              params: np.ndarray,
                              n_perturbations: int = 20) -> np.ndarray:
        """
        Quantum analogue of SHAP: estimate feature attributions by
        perturbing each input feature and measuring output change.

        For feature i:
            attr_i = E[|f(x) - f(x with x_i = random)|]

        Returns:
            attributions: Array of shape (n_qubits,), normalized to [0,1]
        """
        @qml.qnode(self.dev, interface="numpy")
        def circuit(x, p):
            for i in range(self.n_qubits):
                qml.RY(np.pi * float(x[i]), wires=i)
            qml.StronglyEntanglingLayers(p, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))

        baseline = float(circuit(X_sample, params))
        attributions = np.zeros(self.n_qubits)
        rng = np.random.default_rng(42)

        for feat_idx in range(self.n_qubits):
            deltas = []
            for _ in range(n_perturbations):
                x_perturbed = X_sample.copy()
                x_perturbed[feat_idx] = rng.uniform(0, 1)
                perturbed_out = float(circuit(x_perturbed, params))
                deltas.append(abs(baseline - perturbed_out))
            attributions[feat_idx] = np.mean(deltas)

        # Normalize to [0, 1]
        max_attr = attributions.max()
        if max_attr > 0:
            attributions /= max_attr

        return attributions

    # ─────────────────────────────────────────────────────────────────────
    # 3. BARREN PLATEAU DETECTOR
    # ─────────────────────────────────────────────────────────────────────

    def barren_plateau_analysis(self, X_sample: np.ndarray,
                                 n_random_inits: int = 30,
                                 param_shape: tuple = None) -> dict:
        """
        Detect barren plateaus by measuring gradient variance across
        random parameter initializations.

        In a barren plateau:  Var[∂L/∂θ] → 0 exponentially with n_qubits.
        Healthy training:     Var[∂L/∂θ] > threshold (~ 1e-3)

        Returns dict with per-parameter gradient variance statistics.
        """
        if param_shape is None:
            param_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=self.n_layers, n_wires=self.n_qubits
            )

        @qml.qnode(self.dev, interface="numpy")
        def circuit(x, p_flat):
            p = p_flat.reshape(param_shape)
            for i in range(self.n_qubits):
                qml.RY(np.pi * float(x[i]), wires=i)
            qml.StronglyEntanglingLayers(p, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))

        n_params = int(np.prod(param_shape))
        all_grads = np.zeros((n_random_inits, n_params))
        rng = np.random.default_rng(0)

        print(f"  [QAI] Computing barren plateau analysis ({n_random_inits} random inits)...")

        for trial in range(n_random_inits):
            flat_p = rng.uniform(-np.pi, np.pi, n_params)
            grads = np.zeros(n_params)
            for k in range(n_params):
                p_plus  = flat_p.copy(); p_plus[k]  += np.pi / 2
                p_minus = flat_p.copy(); p_minus[k] -= np.pi / 2
                grads[k] = (circuit(X_sample, p_plus) -
                            circuit(X_sample, p_minus)) / 2.0
            all_grads[trial] = grads

        grad_variance = np.var(all_grads, axis=0)
        mean_variance = float(np.mean(grad_variance))
        threshold = 1e-3

        result = {
            "mean_gradient_variance":  mean_variance,
            "max_gradient_variance":   float(np.max(grad_variance)),
            "min_gradient_variance":   float(np.min(grad_variance)),
            "barren_plateau_detected": mean_variance < threshold,
            "threshold":               threshold,
            "per_param_variance":      grad_variance,
            "n_qubits":                self.n_qubits,
            "n_layers":                self.n_layers,
        }

        status = "[WARN] BARREN PLATEAU" if result["barren_plateau_detected"] else "[OK] HEALTHY GRADIENTS"
        print(f"  [QAI] Mean gradient variance: {mean_variance:.6f} -> {status}")

        return result

    # ─────────────────────────────────────────────────────────────────────
    # 4. ENTANGLEMENT ENTROPY MONITOR
    # ─────────────────────────────────────────────────────────────────────

    def entanglement_entropy(self, X_sample: np.ndarray,
                              params: np.ndarray) -> dict:
        """
        Compute bipartite von Neumann entanglement entropy of the
        quantum state produced by the circuit for a given input.

        S(ρ_A) = -Tr(ρ_A log ρ_A)

        Partition: first n_qubits//2 qubits vs rest.
        S > 0 confirms the circuit is creating genuine quantum entanglement.
        S_max = log(2^(n/2)) = (n/2) * log(2)

        Returns dict with entropy value and normalized entropy.
        """
        dev_statevec = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev_statevec, interface="numpy")
        def statevec_circuit(x, p):
            for i in range(self.n_qubits):
                qml.RY(np.pi * float(x[i]), wires=i)
            qml.StronglyEntanglingLayers(p, wires=range(self.n_qubits))
            return qml.state()

        state = statevec_circuit(X_sample, params)
        n = self.n_qubits
        dim = 2 ** n
        n_A = n // 2
        dim_A = 2 ** n_A
        dim_B = 2 ** (n - n_A)

        # Reshape state into bipartite system
        psi = state.reshape(dim_A, dim_B)
        # Singular value decomposition to get Schmidt coefficients
        singular_vals = np.linalg.svd(psi, compute_uv=False)
        # Probability amplitudes squared
        probs = singular_vals ** 2
        probs = probs[probs > 1e-15]  # filter numerical zeros
        # Von Neumann entropy
        entropy = float(-np.sum(probs * np.log2(probs + 1e-15)))
        max_entropy = n_A * np.log2(2)
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            "von_neumann_entropy":  entropy,
            "normalized_entropy":   normalized,
            "max_possible_entropy": max_entropy,
            "n_qubits":             n,
            "partition":            f"[0..{n_A-1}] vs [{n_A}..{n-1}]",
            "is_entangled":         entropy > 0.01,
        }

    # ─────────────────────────────────────────────────────────────────────
    # 5. QUANTUM DECISION EXPLANATION
    # ─────────────────────────────────────────────────────────────────────

    def explain_prediction(self, X_sample_raw: np.ndarray,
                            feature_names: list,
                            params: np.ndarray,
                            preprocessor) -> dict:
        """
        Generate a human-readable explanation for a single prediction.
        Combines feature attributions + entanglement + prediction confidence.

        Returns dict with full explanation suitable for clinical display.
        """
        # Preprocess using model's pipeline
        X_proc = preprocessor(X_sample_raw)

        # Prediction probability
        prob_dementia = float((1 + self._circuit_eval(X_proc, params)) / 2)
        pred_class = 1 if prob_dementia >= 0.5 else 0
        confidence = max(prob_dementia, 1 - prob_dementia)

        # Feature attribution
        attributions = self.feature_attribution(X_proc, params)

        # Entanglement
        ent_info = self.entanglement_entropy(X_proc, params)

        # Build explanation
        attr_with_names = sorted(
            zip(feature_names[:self.n_qubits], attributions),
            key=lambda x: x[1], reverse=True
        )

        explanation = {
            "prediction":         "Dementia Risk" if pred_class == 1 else "Low Risk",
            "confidence":         round(confidence * 100, 1),
            "probability_dementia": round(prob_dementia * 100, 1),
            "top_features":       [(name, round(float(attr), 3))
                                   for name, attr in attr_with_names[:3]],
            "entanglement_entropy": round(ent_info["von_neumann_entropy"], 4),
            "normalized_entropy":   round(ent_info["normalized_entropy"], 4),
            "quantum_correlations": "Present" if ent_info["is_entangled"] else "Minimal",
            "circuit_depth":        self.n_layers * self.n_qubits,
        }
        return explanation

    def _circuit_eval(self, x: np.ndarray, params: np.ndarray) -> float:
        """Evaluate VQC circuit on a preprocessed sample."""
        @qml.qnode(self.dev, interface="numpy")
        def circuit(x, p):
            for i in range(self.n_qubits):
                qml.RY(np.pi * float(x[i]), wires=i)
            qml.StronglyEntanglingLayers(p, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))
        return float(circuit(x, params))


def print_explanation(explanation: dict):
    """Pretty-print a quantum explanation to the console."""
    border = "=" * 55
    print(f"\n+{border}+")
    print(f"|  QUANTUM DEMENTIA RISK ASSESSMENT                   |")
    print(f"+{border}+")
    print(f"|  Prediction : {explanation['prediction']:<38} |")
    print(f"|  Confidence : {explanation['confidence']}%{' '*36}|")
    print(f"|  P(Dementia): {explanation['probability_dementia']}%{' '*36}|")
    print(f"+{border}+")
    print(f"|  TOP CONTRIBUTING FEATURES (Quantum Attribution)    |")
    for name, attr in explanation["top_features"]:
        bar = "#" * int(attr * 20)
        print(f"|    {name:<12}: {bar:<20} ({attr:.3f})        |")
    print(f"+{border}+")
    print(f"|  QUANTUM STATE ANALYSIS                             |")
    print(f"|  Entanglement Entropy : {explanation['entanglement_entropy']:<26} |")
    print(f"|  Normalized Entropy   : {explanation['normalized_entropy']:<26} |")
    print(f"|  Quantum Correlations : {explanation['quantum_correlations']:<26} |")
    print(f"+{border}+")
