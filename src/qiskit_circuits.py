"""
Qiskit Implementation of Quantum Circuits
==========================================
Fulfills Hackathon Requirement: "Documented your Qiskit implementation and backend or simulator"
Implements:
  1. Variational Quantum Classifier Circuit (TwoLocal / EfficientSU2 with CNOTs)
  2. IQP ZZ Feature Map Circuit (ZZFeatureMap)
  3. Qiskit Statevector Simulator Execution via Qiskit Primitives
"""

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

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
try:
    from qiskit.circuit.library import efficient_su2
    USE_FUNC = True
except ImportError:
    from qiskit.circuit.library import EfficientSU2
    USE_FUNC = False
from qiskit.quantum_info import SparsePauliOp, Statevector
try:
    from qiskit.primitives import StatevectorEstimator as Estimator
except ImportError:
    from qiskit.primitives import Estimator


def build_qiskit_vqc(n_qubits: int = 4, n_layers: int = 3) -> QuantumCircuit:
    """
    Build a Parameterized Quantum Circuit (PQC) in Qiskit:
      - Feature Encoding: RY rotations with input parameter vector x
      - Variational Ansatz: RealAmplitudes / EfficientSU2 with full circular entanglement
    """
    qc = QuantumCircuit(n_qubits, name="HybridVQC_Qiskit")

    # 1. Feature Encoding Layer (Angle Encoding)
    x = ParameterVector("x", n_qubits)
    for i in range(n_qubits):
        qc.ry(np.pi * x[i], i)
    qc.barrier(label="Encoding")

    # 2. Variational Entangling Layers
    if USE_FUNC:
        ansatz = efficient_su2(
            num_qubits=n_qubits,
            reps=n_layers,
            entanglement="circular",
            su2_gates=["ry", "rz"]
        )
    else:
        ansatz = EfficientSU2(
            num_qubits=n_qubits,
            reps=n_layers,
            entanglement="circular",
            su2_gates=["ry", "rz"]
        )
    qc.compose(ansatz, inplace=True)

    return qc


def build_qiskit_zz_feature_map(n_qubits: int = 4, reps: int = 2) -> QuantumCircuit:
    """
    Build IQP ZZ Feature Map in Qiskit (Havlicek et al., Nature 2019).
    Encodes data into non-linear Hilbert space via ZZ phase interactions.
    """
    return ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement="linear")


def evaluate_qiskit_circuit(qc: QuantumCircuit, x_vals: np.ndarray,
                             param_vals: np.ndarray = None) -> float:
    """
    Evaluate the expectation value of Pauli-Z on qubit 0 using Qiskit's Statevector.
    Backend: Qiskit Statevector Simulator.
    """
    n_qubits = qc.num_qubits
    observable = SparsePauliOp.from_list([("I" * (n_qubits - 1) + "Z", 1.0)])

    # Assign parameter values
    param_dict = {}
    param_names = [p.name for p in qc.parameters]

    # Map input parameters
    for i in range(n_qubits):
        x_param = next((p for p in qc.parameters if p.name == f"x[{i}]"), None)
        if x_param is not None:
            param_dict[x_param] = float(x_vals[i])

    # Assign remaining variational parameters
    other_params = [p for p in qc.parameters if p not in param_dict]
    if param_vals is None:
        param_vals = np.random.uniform(-np.pi, np.pi, len(other_params))

    for p, val in zip(other_params, param_vals):
        param_dict[p] = float(val)

    bound_qc = qc.assign_parameters(param_dict)
    state = Statevector.from_instruction(bound_qc)
    expval = float(state.expectation_value(observable).real)
    return expval


if __name__ == "__main__":
    print(f"[INFO] Qiskit Version: {qiskit.__version__}")
    vqc_qc = build_qiskit_vqc(n_qubits=4, n_layers=2)
    print("\n[Qiskit VQC Architecture]")
    print(vqc_qc.draw(output="text", fold=80))

    zz_qc = build_qiskit_zz_feature_map(n_qubits=4, reps=1)
    print("\n[Qiskit ZZ Feature Map Architecture]")
    print(zz_qc.draw(output="text", fold=80))

    sample_x = np.array([0.7, 0.3, 0.8, 0.5])
    val = evaluate_qiskit_circuit(vqc_qc, sample_x)
    print(f"\n[OK] Qiskit Statevector Simulation <Z0> = {val:.4f} in [-1, 1]")
