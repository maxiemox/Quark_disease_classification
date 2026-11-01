"""
Quantum Circuit Definitions for Hybrid QML Pipeline
=====================================================

Implements two quantum encoding strategies:
  1. Angle Encoding (AmplitudeEmbedding-style via RY rotations)
  2. IQP-style ZZ Feature Map (similar to Qiskit's ZZFeatureMap)

And two variational ansatz circuits:
  1. Strongly Entangling Layers (StronglyEntanglingLayers)
  2. Efficient SU2 (Hardware-Efficient Ansatz)

The quantum kernel function used in QSVM is also defined here.
"""

import pennylane as qml
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE SETUP
# ─────────────────────────────────────────────────────────────────────────────

def make_device(n_qubits: int, backend: str = "lightning.qubit"):
    """Create a PennyLane device. Falls back to default.qubit if lightning unavailable."""
    try:
        dev = qml.device(backend, wires=n_qubits)
        return dev
    except Exception:
        return qml.device("default.qubit", wires=n_qubits)


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENCODING CIRCUITS
# ─────────────────────────────────────────────────────────────────────────────

def angle_encoding(x: np.ndarray, n_qubits: int):
    """
    Angle (basis) encoding via RY rotations.
    Maps each feature to a qubit rotation angle: θ_i = π * x_i (x ∈ [0,1]).
    """
    for i in range(n_qubits):
        qml.RY(np.pi * x[i], wires=i)


def iqp_feature_map(x: np.ndarray, n_qubits: int, reps: int = 2):
    """
    IQP-style ZZ Feature Map (Havlíček et al., 2019).
    Encodes data using ZZ interactions, creating non-linear feature maps in Hilbert space.
    This is the key encoding used in Quantum Kernel SVM.

    Circuit depth: O(reps * n_qubits^2)
    """
    for _ in range(reps):
        # Hadamard layer
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        # Single-qubit Z-phase rotations
        for i in range(n_qubits):
            qml.PhaseShift(2.0 * x[i], wires=i)

        # Two-qubit ZZ-phase entanglement
        for i in range(n_qubits - 1):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])
                qml.PhaseShift(2.0 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                qml.CNOT(wires=[i, j])


# ─────────────────────────────────────────────────────────────────────────────
#  VARIATIONAL ANSATZ CIRCUITS
# ─────────────────────────────────────────────────────────────────────────────

def strongly_entangling_ansatz(params: np.ndarray, n_qubits: int, n_layers: int):
    """
    StronglyEntanglingLayers ansatz.
    params shape: (n_layers, n_qubits, 3) — three rotation angles per qubit per layer.
    """
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))


def hardware_efficient_ansatz(params: np.ndarray, n_qubits: int, n_layers: int):
    """
    Hardware-Efficient Ansatz (Efficient SU2 style).
    Alternating Ry-Rz rotations + CNOT entanglement.
    params shape: (n_layers, n_qubits, 2)
    """
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[layer, i, 0], wires=i)
            qml.RZ(params[layer, i, 1], wires=i)
        # Linear entanglement
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Periodic entanglement (wrap-around)
        if n_qubits > 2:
            qml.CNOT(wires=[n_qubits - 1, 0])


# ─────────────────────────────────────────────────────────────────────────────
#  VARIATIONAL QUANTUM CLASSIFIER (VQC)
# ─────────────────────────────────────────────────────────────────────────────

def build_vqc(n_qubits: int, n_layers: int, ansatz: str = "strongly_entangling",
              device=None):
    """
    Build a VQC QNode that returns expectation value of PauliZ on qubit 0.

    Args:
        n_qubits:  Number of qubits (= number of input features)
        n_layers:  Depth of variational circuit
        ansatz:    'strongly_entangling' | 'hardware_efficient'
        device:    PennyLane device (defaults to default.qubit)

    Returns:
        qnode (callable), param_shape (tuple)
    """
    if device is None:
        device = make_device(n_qubits)

    if ansatz == "strongly_entangling":
        param_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    else:
        param_shape = (n_layers, n_qubits, 2)

    @qml.qnode(device, interface="autograd")
    def circuit(x, params):
        # 1. Feature encoding
        angle_encoding(x, n_qubits)
        # 2. Variational ansatz
        if ansatz == "strongly_entangling":
            strongly_entangling_ansatz(params, n_qubits, n_layers)
        else:
            hardware_efficient_ansatz(params, n_qubits, n_layers)
        # 3. Measurement
        return qml.expval(qml.PauliZ(0))

    return circuit, param_shape


# ─────────────────────────────────────────────────────────────────────────────
#  QUANTUM KERNEL (for QSVM)
# ─────────────────────────────────────────────────────────────────────────────

def build_quantum_kernel(n_qubits: int, encoding: str = "iqp", device=None):
    """
    Build a quantum kernel function using fidelity-based overlap:
        K(x1, x2) = |⟨φ(x1)|φ(x2)⟩|²

    This implements the kernel trick in quantum Hilbert space — the foundation
    of Quantum Support Vector Machines (QSVM).

    Args:
        n_qubits: Number of qubits
        encoding: 'iqp' (ZZFeatureMap) or 'angle'
        device:   PennyLane device

    Returns:
        kernel_fn(x1, x2) → float in [0, 1]
    """
    if device is None:
        device = make_device(n_qubits)

    @qml.qnode(device, interface="numpy")
    def kernel_circuit(x1, x2):
        if encoding == "iqp":
            iqp_feature_map(x1, n_qubits, reps=2)
        else:
            angle_encoding(x1, n_qubits)

        if encoding == "iqp":
            qml.adjoint(iqp_feature_map)(x2, n_qubits, reps=2)
        else:
            qml.adjoint(angle_encoding)(x2, n_qubits)

        return qml.probs(wires=range(n_qubits))

    def kernel_fn(x1: np.ndarray, x2: np.ndarray) -> float:
        probs = kernel_circuit(x1, x2)
        return float(probs[0])  # |⟨0|U†(x2)U(x1)|0⟩|²

    return kernel_fn


def build_kernel_matrix(X: np.ndarray, kernel_fn, batch_size: int = None) -> np.ndarray:
    """
    Compute the full NxN kernel matrix K[i,j] = kernel_fn(X[i], X[j]).
    For large datasets, processes in batches with progress output.
    """
    n = len(X)
    K = np.zeros((n, n))

    print(f"[INFO] Computing {n}x{n} quantum kernel matrix...")
    total = n * (n + 1) // 2

    count = 0
    for i in range(n):
        for j in range(i, n):
            val = kernel_fn(X[i], X[j])
            K[i, j] = val
            K[j, i] = val
            count += 1
            if count % 50 == 0:
                pct = 100 * count / total
                print(f"  Progress: {count}/{total} ({pct:.1f}%)")

    print(f"[INFO] Kernel matrix complete. Shape: {K.shape}")
    return K


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY: Circuit drawer
# ─────────────────────────────────────────────────────────────────────────────

def draw_vqc(n_qubits: int = 4, n_layers: int = 2):
    """Print an ASCII representation of the VQC circuit for a given config."""
    dev = make_device(n_qubits)
    circuit, param_shape = build_vqc(n_qubits, n_layers, device=dev)

    x_sample = np.random.uniform(0, 1, n_qubits)
    params_sample = np.random.uniform(-np.pi, np.pi, param_shape)

    drawer = qml.draw(circuit)
    print(drawer(x_sample, params_sample))


def draw_kernel_circuit(n_qubits: int = 4):
    """Draw the quantum kernel circuit."""
    dev = make_device(n_qubits)
    kernel_fn_raw = build_quantum_kernel(n_qubits, encoding="iqp", device=dev)

    # Re-expose the qnode for drawing
    @qml.qnode(dev, interface="numpy")
    def kernel_circuit(x1, x2):
        iqp_feature_map(x1, n_qubits, reps=2)
        qml.adjoint(iqp_feature_map)(x2, n_qubits, reps=2)
        return qml.probs(wires=range(n_qubits))

    x1 = np.random.uniform(0, np.pi, n_qubits)
    x2 = np.random.uniform(0, np.pi, n_qubits)
    print(qml.draw(kernel_circuit)(x1, x2))


if __name__ == "__main__":
    print("=== VQC Circuit ===")
    draw_vqc(n_qubits=4, n_layers=2)
    print("\n=== Quantum Kernel Circuit ===")
    draw_kernel_circuit(n_qubits=4)
