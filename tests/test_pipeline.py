"""
Test suite for QuantumX pipeline.
Run with: python tests/test_pipeline.py
"""

import sys, os
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import unittest


class TestDataParser(unittest.TestCase):
    def setUp(self):
        self.disc_root = r"e:\New folder\Downloads\QuantumX\oasis_cross-sectional_disc1"

    def test_load_dataset(self):
        from src.data_parser import load_oasis_dataset
        df = load_oasis_dataset(self.disc_root)
        self.assertGreater(len(df), 0, "Dataset should not be empty")
        self.assertIn("CDR", df.columns)
        self.assertIn("AGE", df.columns)
        self.assertIn("nWBV", df.columns)
        print(f"  ✅ Dataset loaded: {len(df)} records")

    def test_preprocess(self):
        from src.data_parser import load_oasis_dataset, preprocess
        df = load_oasis_dataset(self.disc_root)
        X, y = preprocess(df, binary=True)
        self.assertFalse(X.isnull().any().any(), "X should have no NaNs")
        self.assertEqual(y.max(), 1, "Binary labels should be 0 or 1")
        self.assertEqual(y.min(), 0)
        print(f"  ✅ Preprocessing OK: {X.shape}, y unique: {y.unique()}")


class TestQuantumCircuits(unittest.TestCase):
    def setUp(self):
        import pennylane as qml
        self.n_qubits = 4
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

    def test_angle_encoding(self):
        import pennylane as qml
        from src.quantum_circuits import angle_encoding

        @qml.qnode(self.dev)
        def circuit(x):
            angle_encoding(x, self.n_qubits)
            return qml.state()

        x = np.random.uniform(0, 1, self.n_qubits)
        state = circuit(x)
        self.assertEqual(len(state), 2 ** self.n_qubits)
        print(f"  ✅ Angle encoding: state dim = {len(state)}")

    def test_iqp_feature_map(self):
        import pennylane as qml
        from src.quantum_circuits import iqp_feature_map

        @qml.qnode(self.dev)
        def circuit(x):
            iqp_feature_map(x, self.n_qubits, reps=1)
            return qml.probs(wires=range(self.n_qubits))

        x = np.random.uniform(0, np.pi, self.n_qubits)
        probs = circuit(x)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        print(f"  ✅ IQP feature map: probs sum = {np.sum(probs):.6f}")

    def test_vqc_build(self):
        from src.quantum_circuits import build_vqc
        circuit, param_shape = build_vqc(self.n_qubits, n_layers=2)
        x = np.random.uniform(0, 1, self.n_qubits)
        params = np.random.uniform(-np.pi, np.pi, param_shape)
        out = circuit(x, params)
        self.assertTrue(-1.0 <= float(out) <= 1.0, "Output should be in [-1, 1]")
        print(f"  [PASS] VQC output: {float(out):.4f} in [-1, 1]")


class TestHybridVQC(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import make_classification
        self.X, self.y = make_classification(
            n_samples=30, n_features=9, n_informative=5, random_state=42
        )
        self.X = self.X.astype(float)

    def test_fit_predict(self):
        from src.hybrid_vqc import HybridVQC
        vqc = HybridVQC(n_qubits=3, n_layers=2, max_iter=5, verbose=False)
        vqc.fit(self.X, self.y)
        self.assertTrue(vqc.is_fitted)

        preds = vqc.predict(self.X)
        self.assertEqual(len(preds), len(self.X))
        self.assertTrue(set(preds).issubset({0, 1}))

        proba = vqc.predict_proba(self.X)
        self.assertEqual(proba.shape, (len(self.X), 2))
        print(f"  ✅ VQC fit+predict: {len(preds)} predictions, "
              f"accuracy={np.mean(preds == self.y):.3f}")

    def test_n_parameters(self):
        from src.hybrid_vqc import HybridVQC
        vqc = HybridVQC(n_qubits=4, n_layers=3, verbose=False)
        n_params = vqc.n_parameters()
        self.assertGreater(n_params, 0)
        print(f"  ✅ VQC parameters: {n_params}")

    def test_circuit_diagram(self):
        from src.hybrid_vqc import HybridVQC
        vqc = HybridVQC(n_qubits=3, n_layers=2, verbose=False)
        diagram = vqc.get_circuit_diagram()
        self.assertIsInstance(diagram, str)
        self.assertGreater(len(diagram), 0)
        print(f"  ✅ Circuit diagram generated ({len(diagram)} chars)")


class TestClassicalBaseline(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import make_classification
        self.X, self.y = make_classification(
            n_samples=40, n_features=9, n_informative=5, random_state=42
        )

    def test_run_baseline(self):
        from src.classical_baseline import run_classical_baseline
        results = run_classical_baseline(self.X, self.y, cv=3)
        self.assertFalse(results.empty)
        self.assertIn("accuracy_mean", results.columns)
        print(f"  ✅ Classical baseline: {len(results)} models evaluated")

    def test_feature_importance(self):
        from src.classical_baseline import get_feature_importance
        names = [f"f{i}" for i in range(9)]
        imp = get_feature_importance(self.X, self.y, names)
        self.assertEqual(len(imp), 9)
        self.assertTrue((imp["importance"] >= 0).all())
        print(f"  ✅ Feature importance: top = {imp.iloc[0]['feature']}")


if __name__ == "__main__":
    print("\n" + "="*55)
    print("   QuantumX Test Suite")
    print("="*55 + "\n")

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # Add all test classes
    for cls in [TestDataParser, TestQuantumCircuits,
                TestHybridVQC, TestClassicalBaseline]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n[PASS] All tests passed!")
    else:
        print(f"\n[FAIL] {len(result.failures)} failures, {len(result.errors)} errors")
        sys.exit(1)
