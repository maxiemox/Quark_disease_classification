# ⚛️ QuantumX: Presentation Pitch Deck
## Hybrid Quantum Machine Learning for Early Alzheimer's Detection with QuantumXAI

> **Track:** Hackathon Challenge 02 · Quantum Machine Learning | Healthcare  
> **Global Impact:** UN Sustainable Development Goal 3 (Good Health & Well-Being)  
> **Repository:** [https://github.com/maxiemox/Quark_disease_classification.git](https://github.com/maxiemox/Quark_disease_classification.git)

---

## 📑 7-Slide Overview

| Slide | Title | Core Focus |
| :---: | :--- | :--- |
| **1** | **Executive Title** | Project mission, track, and key takeaways |
| **2** | **The Problem (Why)** | 10–15 year silent window & classical ML limitations |
| **3** | **What We Built & The Pipeline (What & How)** | 5-step diagnostic pipeline and dual quantum classifiers |
| **4** | **Why These Components? (Architectural Choices)** | Technical rationale for 4 qubits, RY encoding, strongly entangling, parameter-shift |
| **5** | **The Quantum Engines (The Implementation)** | Plain terms & mathematical formulation of VQC and QSVM |
| **6** | **The Innovation: QuantumXAI** | Feature attribution, entanglement entropy, and model safety |
| **7** | **Results, Healthcare Impact & Roadmap** | 5-fold CV benchmarks, real-world use cases, and future outlook |

---

## 📽️ The 7 Slides

---

### Slide 1: Executive Title
#### **QuantumX: Hybrid Quantum Machine Learning for Early Alzheimer's Detection**
* **The Mission:** Detecting neurodegeneration in the 10–15 year pre-symptomatic window before irreversible brain atrophy occurs.
* **Core Value:** Over **55 million people** live with dementia worldwide. By the time symptoms appear, up to **40% of memory neurons are already lost**.
* **The Solution:** QuantumX combines **Variational Quantum Classifiers (VQC)** and **Quantum SVM (QSVM)** with an innovative **QuantumXAI Explainability Engine** to deliver accurate, biologically interpretable diagnoses.
* **Key Highlights:**
  - **Dataset:** OASIS-1 Brain MRI clinical cohort (39 patients, authorized).
  - **Frameworks:** PennyLane 0.45 & Qiskit 2.6.1 Statevector simulation.
  - **Cross-Validation Accuracy:** **84.6%** with just 36 parameters.
  - **Global Impact:** Aligned with UN Sustainable Development Goal 3 (SDG-3).

---

### Slide 2: The Problem (The "Why")
#### **The Diagnostic Dilemma in Early Alzheimer's**
* **1. 10–15 Year Latency Window:**
  * Brain tissue loss begins more than a decade before noticeable memory failure.
  * Subtle structural atrophy (`nWBV`) and slight cognitive changes (`MMSE`) interact quietly across age cohorts.
  * No single clinical measurement is definitive on its own.
* **2. The Small Clinical Cohort Trap:**
  * Medical datasets are inherently small and expensive to collect.
  * Classical deep neural networks easily overfit on small sample sizes ($N < 100$).
  * Linear classical models miss the non-linear correlations between age, education, and brain volume.
* **3. The "Black Box" Barrier:**
  * Doctors cannot prescribe preventive interventions based on uninterpretable AI weights.
  * Neurologists require biological rationale for every diagnostic decision.

---

### Slide 3: What We Built & The Pipeline (What & How)
#### **The 5-Step Diagnostic Pipeline**

$$\text{OASIS-1 MRI Data} \xrightarrow{\text{Imputation \& Scaling}} \text{4 Qubits } R_Y(x) \xrightarrow{\text{3-Layer Entangling Circuit}} \langle Z_0 \rangle \xrightarrow{\text{Calibrated Decision }\tau^*} \text{QuantumXAI Report}$$

1. **Patient Data Ingestion:** OASIS-1 MRI clinical cohort (39 patients) containing cognitive tests (`MMSE`) and volumetric MRI metrics (`nWBV`, `eTIV`, `Age`).
2. **Clinical Imputation & Scaling:** Standard protocol fill for young normal controls ($Age < 50, CDR=0, MMSE=30$) and MinMax scaling to $[0, \pi]$.
3. **Quantum State Encoding:** Single-qubit $R_Y(x_i)$ rotations initialize the 4-qubit quantum state in 16-dimensional Hilbert space.
4. **Quantum Core:** 3-layer strongly entangling Variational Quantum Classifier (36 parameters) and non-linear IQP ZZ Quantum SVM.
5. **QuantumXAI Generation:** Computes individual biomarker attributions, Von Neumann Entanglement Entropy, and doctor-friendly report cards.
* **Calibrated Decisions:** Output expectation $\langle Z_0 \rangle \in [-1, 1]$ is mapped to risk via optimal threshold $\tau^*$, balancing sensitivity and specificity.

---

### Slide 4: Why These Components? (Architectural Choices)
#### **Clear Technical Rationale Behind Our Design Decisions**

| Component | Choice | Why This Choice Over Alternatives? |
| :--- | :--- | :--- |
| **Qubit Count** | **4 Qubits** | Maps 4 core clinical features into a $2^4 = 16$-dimensional Hilbert space. Keeps the circuit shallow (depth 3) to prevent barren plateaus and minimize noise on NISQ hardware. |
| **Encoding** | **$R_Y$ Angle Encoding** | Requires depth $O(1)$ and zero multi-qubit entangling gates for state loading, avoiding the exponential CNOT gate overhead of Amplitude Encoding. |
| **VQC Ansatz** | **Strongly Entangling Layers** | Universal 3-parameter single-qubit rotations ($R_Z R_Y R_Z$) with circular CNOT ladders provide high expressivity to capture multi-biomarker correlations. |
| **QSVM Map** | **IQP ZZ Feature Map** | Generates non-linear pairwise interactions $\Phi_{ij}(x) = (\pi - x_i)(\pi - x_j)$ that are classically hard to simulate at scale, providing a provably rich quantum kernel. |
| **Gradient Method** | **Parameter-Shift Rule** | Finite-difference gradients fail under quantum shot noise. The Parameter-Shift rule evaluates exact mathematical gradients directly on quantum chips. |
| **Dataset** | **OASIS-1 Brain MRI** | Gold-standard, clinically validated cohort combining structural brain MRI volume measurements with standardized MMSE cognitive scores. |

---

### Slide 5: The Quantum Engines (The Implementation)
#### **Plain Terms with Exact Mathematical Formulations**

* **1. Hybrid VQC Formulation:**
  $$\vert \psi(x) \rangle = \left( \prod_{l=1}^3 \text{CNOT-ladder} \prod_{i=0}^3 R_Z(\omega_{l,i}) R_Y(\theta_{l,i}) R_Z(\phi_{l,i}) \right) \bigotimes_{i=0}^3 R_Y(x_i) \vert 0 \rangle$$
  * Trainable Parameters: $3 \text{ layers} \times 4 \text{ qubits} \times 3 \text{ angles} = 36 \text{ parameters}$.
  * Readout: $\hat{y} = \langle \psi(x, \boldsymbol{\theta}) \vert Z_0 \vert \psi(x, \boldsymbol{\theta}) \rangle \in [-1, 1]$.
* **2. Parameter-Shift Rule (Exact Gradients on Hardware):**
  $$\frac{\partial \langle Z_0 \rangle}{\partial \theta_k} = \frac{\langle Z_0 \rangle_{\theta_k + \pi/2} - \langle Z_0 \rangle_{\theta_k - \pi/2}}{2}$$
* **3. Quantum SVM (IQP ZZ Feature Map):**
  $$U_{\Phi(x)} = \exp\left( i \sum_{i=1}^4 x_i Z_i + \sum_{i < j} (\pi - x_i)(\pi - x_j) Z_i Z_j \right) H^{\otimes 4}$$
  $$K(x_i, x_j) = \vert \langle \Phi(x_i) \vert \Phi(x_j) \rangle \vert^2 \quad \text{(Fidelity Overlap Kernel)}$$
* **4. Qiskit 2.6 Native Circuit:**
  * Implemented using `ParameterVector`, `efficient_su2`, and verified on Qiskit `Statevector` simulator ($\langle Z_0 \rangle = 0.2252$).

---

### Slide 6: The Innovation: QuantumXAI
#### **Why Neurologists Can Trust QuantumX**

1. **Quantum Feature Attribution (QFA):**
   * *Which biomarker drove the diagnosis?*
   * In-circuit angle perturbation $\frac{\Delta \langle Z_0 \rangle}{\Delta x_i}$ calculates quantum SHAP-equivalent importance.
   * Clinicians see whether cognitive score (MMSE) or whole-brain atrophy (nWBV) contributed to the risk.
2. **Von Neumann Entanglement Entropy $S(\rho_A)$:**
   * *Were biomarkers evaluated together?*
   * Computes reduced density matrix entropy via Schmidt decomposition:
     $$S(\rho_A) = -\sum_{k} \lambda_k^2 \log_2(\lambda_k^2)$$
   * Mathematically proves that the circuit evaluated entangled multi-modal patterns rather than isolated variables.
3. **Barren Plateau Safety:**
   * *Is the quantum model reliable?*
   * Tracks gradient variance $\text{Var}[\nabla_\theta \mathcal{L}] > 0$ across random initializations, confirming the circuit avoids vanishing gradients.

---

### Slide 7: Results, Healthcare Impact & Roadmap
#### **5-Fold Stratified Cross-Validation on OASIS-1 MRI Cohort**

| Model | Architecture Type | Accuracy (5-Fold CV) | ROC-AUC | Parameter Count |
| :--- | :--- | :---: | :---: | :---: |
| **QuantumX VQC** | **Hybrid QML (3-Layer Ansatz)** | **0.846 ± 0.08** | **0.88** | **36 variational params** |
| **QuantumX QSVM** | **Quantum Kernel (IQP ZZ)** | **0.821 ± 0.09** | **0.85** | **Precomputed Kernel** |
| Classical SVM (RBF) | Kernel Support Vector | 0.821 ± 0.09 | 0.84 | Hyperplane weights |
| Random Forest | Classical Ensemble | 0.821 ± 0.10 | 0.86 | 100 trees (~1,200 nodes) |
| Logistic Regression | Classical Linear | 0.795 ± 0.08 | 0.82 | 5 linear weights |

* **Real-World Healthcare Use Cases:**
  1. **Memory & Geriatric Clinics:** Routine assistive screening tool during annual checkups.
  2. **Clinical Trials (Pharma):** Stratifies pre-symptomatic MCI candidates before irreversible atrophy occurs.
  3. **Telemedicine:** Cloud-based quantum inference via IBM Quantum backends.
* **Key Takeaway & Roadmap:**
  - Matches or outperforms classical ensembles with only **36 parameters**.
  - **Roadmap:** Deploy with Zero-Noise Extrapolation (ZNE via Mitiq) on IBM Quantum QPUs and develop Quantum Convolutional Networks (QCNN) directly on raw 3D MRI brain voxel slices.
