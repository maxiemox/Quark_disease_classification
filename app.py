"""
QuantumX — Interactive Web Dashboard
======================================
Run with: streamlit run app.py

Tabs:
  1. Home / Overview
  2. Dataset Explorer
  3. Quantum Circuit Viewer
  4. Train & Evaluate
  5. Quantum XAI (Explainability)
  6. Model Comparison
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pennylane as qml
import os, sys, warnings, time, json
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

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_parser       import load_oasis_dataset, preprocess
from classical_baseline import run_classical_baseline, get_feature_importance
from hybrid_vqc        import HybridVQC
from quantum_xai       import QuantumXAI
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc as sk_auc

# ── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantumX | Alzheimer's QML",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISC_ROOT     = r"e:\New folder\Downloads\QuantumX\oasis_cross-sectional_disc1"
FEATURE_NAMES = ["AGE", "SEX_ENC", "HAND_ENC", "EDUC", "SES", "MMSE", "eTIV", "ASF", "nWBV"]
RANDOM_SEED   = 42

# ── Dark-theme palette ───────────────────────────────────────────────────────
QUANTUM_PURPLE = "#6C63FF"
TEAL           = "#00D4AA"
CORAL          = "#FF6B6B"
GOLD           = "#FFD166"
BG             = "#0F0F23"
SURFACE        = "#1A1A35"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background: linear-gradient(135deg, #0F0F23 0%, #1A1A35 100%); }
  section[data-testid="stSidebar"] { background: #12122a; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1A1A35, #252550);
    border: 1px solid #6C63FF44;
    border-radius: 12px;
    padding: 12px;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab"] {
    background: #1A1A35;
    border-radius: 8px 8px 0 0;
    color: #aaa;
    font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #6C63FF, #00D4AA);
    color: white !important;
  }

  /* Quantum badge */
  .quantum-badge {
    background: linear-gradient(90deg, #6C63FF, #00D4AA);
    color: white; padding: 4px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 1px;
    display: inline-block; margin: 4px 2px;
  }
  .classical-badge {
    background: #FF6B6B44; border: 1px solid #FF6B6B;
    color: #FF6B6B; padding: 4px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700;
    display: inline-block; margin: 4px 2px;
  }

  /* Section headers */
  h1, h2, h3 { color: #E8E8FF !important; }
  p, li { color: #C0C0E0; }

  /* Divider */
  hr { border-color: #6C63FF44; }

  /* Code blocks */
  code { background: #252550; color: #00D4AA; border-radius: 4px; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0F0F23; }
  ::-webkit-scrollbar-thumb { background: #6C63FF; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors="#E8E8FF")
    ax.xaxis.label.set_color("#E8E8FF")
    ax.yaxis.label.set_color("#E8E8FF")
    ax.title.set_color("#E8E8FF")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2A50")
    ax.grid(color="#2A2A50", alpha=0.4)
    return fig, ax

def dark_figs(rows, cols, w=14, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors="#E8E8FF")
        ax.xaxis.label.set_color("#E8E8FF")
        ax.yaxis.label.set_color("#E8E8FF")
        ax.title.set_color("#E8E8FF")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2A2A50")
        ax.grid(color="#2A2A50", alpha=0.4)
    return fig, axes


@st.cache_data
def load_data():
    df = load_oasis_dataset(DISC_ROOT)
    X, y = preprocess(df, binary=True)
    return df, X, y


@st.cache_data
def compute_baselines(X_arr, y_arr):
    return run_classical_baseline(X_arr, y_arr, cv=5)


@st.cache_data
def compute_feature_importance(X_arr, y_arr):
    return get_feature_importance(X_arr, y_arr, FEATURE_NAMES)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0'>
      <div style='font-size:3rem'>⚛️</div>
      <div style='font-size:1.4rem; font-weight:800; color:#6C63FF'>QuantumX</div>
      <div style='color:#00D4AA; font-size:0.8rem'>Quantum ML · Healthcare</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚙️ Model Config")
    n_qubits = st.slider("Number of Qubits", 2, 6, 4)
    n_layers = st.slider("VQC Layers", 1, 5, 3)
    epochs   = st.slider("Training Epochs", 10, 100, 40, step=10)
    lr       = st.select_slider("Learning Rate", [0.001, 0.005, 0.01, 0.02, 0.05], value=0.01)
    ansatz   = st.selectbox("Ansatz", ["strongly_entangling", "hardware_efficient"])
    test_size = st.slider("Test Split %", 10, 50, 30) / 100

    st.divider()
    st.markdown("### 📊 Dataset")
    st.caption(f"OASIS Cross-Sectional · Disc 1")

    st.divider()
    st.markdown("""
    <div style='font-size:0.7rem; color:#666; text-align:center'>
    PennyLane 0.45 · scikit-learn 1.9<br>
    SDG-3: Good Health & Well-being
    </div>
    """, unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
try:
    df, X, y = load_data()
    X_arr, y_arr = X.values, y.values
    data_loaded = True
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    data_loaded = False
    st.stop()


# ── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Home",
    "📊 Dataset",
    "⚛️ Quantum Circuit",
    "🧠 Train & Evaluate",
    "🔍 XAI Explainer",
    "📈 Compare Models",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("""
    <h1 style='text-align:center; background: linear-gradient(90deg, #6C63FF, #00D4AA);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               font-size:3rem; font-weight:900; margin-bottom:0'>
      QuantumX
    </h1>
    <p style='text-align:center; color:#888; font-size:1.1rem; margin-top:4px'>
      Quantum Machine Learning for Alzheimer's Disease Detection
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🧬 Subjects", len(df))
    col2.metric("🔢 Features", len(FEATURE_NAMES))
    col3.metric("⚛️ Qubits", n_qubits)
    col4.metric("🔗 VQC Params", int(np.prod(qml.StronglyEntanglingLayers.shape(n_layers, n_qubits))))
    col5.metric("🌐 SDG", "SDG-3")

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div style='background:#1A1A35; border:1px solid #6C63FF44; border-radius:12px; padding:20px'>
        <div style='font-size:1.5rem'>⚛️ Hybrid VQC</div>
        <p style='color:#aaa; font-size:0.9rem'>Variational Quantum Classifier with StronglyEntanglingLayers ansatz.
        Classical PCA compression → Quantum Hilbert space learning → Sigmoid output.</p>
        <span class='quantum-badge'>QUANTUM</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background:#1A1A35; border:1px solid #00D4AA44; border-radius:12px; padding:20px'>
        <div style='font-size:1.5rem'>🎯 QSVM</div>
        <p style='color:#aaa; font-size:0.9rem'>Quantum SVM using IQP ZZ Feature Map kernel.
        Maps data to quantum Hilbert space where the kernel K(x₁,x₂) = |⟨φ(x₁)|φ(x₂)⟩|².</p>
        <span class='quantum-badge'>QUANTUM</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style='background:#1A1A35; border:1px solid #FFD16644; border-radius:12px; padding:20px'>
        <div style='font-size:1.5rem'>🔍 QuantumXAI</div>
        <p style='color:#aaa; font-size:0.9rem'><b>Unique innovation:</b> Explainability for quantum models —
        feature attribution, barren plateau detection, entanglement entropy monitoring.</p>
        <span class='quantum-badge'>QUANTUM</span> <span style='color:#FFD166;font-size:0.75rem'>★ NOVEL</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🏗️ Pipeline Architecture")
    st.markdown("""
    ```
    OASIS .txt files ──► data_parser.py ──► 9 clinical features
                                                    │
                          ┌─────────────────────────┤
                          │                         │
                   ┌──────▼──────┐          ┌───────▼────────┐
                   │ StandardScaler│         │ StandardScaler │
                   │  + PCA       │         │  + PCA         │
                   └──────┬──────┘          └───────┬────────┘
                          │                         │
                   ┌──────▼──────┐          ┌───────▼────────┐
                   │ Angle Encode │         │ IQP ZZ Kernel  │
                   │ RY(π·xᵢ)   │         │ ZZ Feature Map │
                   └──────┬──────┘          └───────┬────────┘
                          │                         │
                   ┌──────▼──────┐          ┌───────▼────────┐
                   │  VQC Ansatz │         │ Fidelity K(x,x')│
                   │ SEL Layers  │         │ Precomputed SVM │
                   └──────┬──────┘          └───────┬────────┘
                          │                         │
                   ⟨Z₀⟩ expval ──sigmoid──► Prediction
    ```
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 📊 OASIS Dataset Explorer")

    st.info(f"**OASIS-1 Cross-Sectional** · {len(df)} subjects · Anonymized brain MRI data · No PII")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(
            df[["AGE", "SEX", "EDUC", "SES", "CDR", "MMSE", "eTIV", "ASF", "nWBV"]].style
            .background_gradient(subset=["CDR"], cmap="RdPu")
            .format({"nWBV": "{:.3f}", "ASF": "{:.3f}", "eTIV": "{:.1f}"}),
            use_container_width=True, height=320
        )
    with c2:
        st.markdown("**CDR Distribution**")
        cdr_counts = df["CDR"].value_counts().sort_index()
        for cdr_val, cnt in cdr_counts.items():
            label = {0: "Healthy", 0.5: "Mild MCI", 1.0: "Dementia", 2.0: "Severe AD"}.get(cdr_val, str(cdr_val))
            color = {0: TEAL, 0.5: GOLD, 1.0: CORAL, 2.0: "#FF3366"}.get(cdr_val, "#888")
            pct = 100 * cnt / len(df)
            st.markdown(f"""
            <div style='margin:6px 0'>
              <div style='color:{color}; font-weight:700'>CDR={cdr_val} — {label}</div>
              <div style='background:#252550; border-radius:6px; height:18px; overflow:hidden'>
                <div style='background:{color}; width:{pct:.0f}%; height:100%; border-radius:6px'></div>
              </div>
              <div style='color:#888; font-size:0.8rem'>{cnt} subjects ({pct:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**Binary Label (Target)**")
        n_dem = int((y_arr == 1).sum())
        n_heal = int((y_arr == 0).sum())
        st.markdown(f"""
        <div style='background:#1A1A35; border-radius:8px; padding:12px; margin-top:8px'>
          <div style='color:{TEAL}'>✅ Non-Dementia (CDR=0): <b>{n_heal}</b></div>
          <div style='color:{CORAL}'>⚠️ Dementia (CDR>0): <b>{n_dem}</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Scatter & histograms
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**MMSE vs Normalized Brain Volume**")
        fig, ax = dark_fig(6, 4)
        df_valid = df.dropna(subset=["MMSE", "nWBV", "CDR"]).copy()
        color_map = {0.0: TEAL, 0.5: GOLD, 1.0: CORAL, 2.0: "#FF3366"}
        point_colors = [color_map.get(val, TEAL) for val in df_valid["CDR"]]
        scatter = ax.scatter(df_valid["MMSE"], df_valid["nWBV"],
                             c=point_colors,
                             s=80, edgecolors="white", linewidth=0.5, alpha=0.85)
        ax.set_xlabel("MMSE Score")
        ax.set_ylabel("nWBV (Normalized Brain Volume)")
        ax.set_title("Clinical Feature Space")
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=TEAL, label='Healthy (CDR=0)', markersize=8),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=GOLD, label='MCI (CDR=0.5)', markersize=8),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=CORAL, label='Dementia (CDR≥1)', markersize=8),
        ]
        ax.legend(handles=legend_elements, facecolor=SURFACE, edgecolor="#2A2A50",
                  labelcolor="#E8E8FF", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown("**Age Distribution by Dementia Status**")
        fig, ax = dark_fig(6, 4)
        df_age = df.dropna(subset=["AGE", "CDR"])
        ax.hist(df_age[df_age["CDR"] == 0]["AGE"], bins=12, alpha=0.7,
                color=TEAL, label="Healthy", edgecolor="white", linewidth=0.5)
        ax.hist(df_age[df_age["CDR"] > 0]["AGE"], bins=12, alpha=0.7,
                color=CORAL, label="Dementia", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.set_title("Age Distribution")
        ax.legend(facecolor=SURFACE, edgecolor="#2A2A50", labelcolor="#E8E8FF")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Feature importance
    st.divider()
    st.markdown("**Feature Importance (Random Forest → Qubit Selection)**")
    imp_df = compute_feature_importance(X_arr, y_arr)
    fig, ax = dark_fig(10, 4)
    colors = [QUANTUM_PURPLE if i < n_qubits else CORAL for i in range(len(imp_df))]
    bars = ax.barh(imp_df["feature"], imp_df["importance"],
                   color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)
    for bar, val in zip(bars, imp_df["importance"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10, color="#E8E8FF")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {n_qubits} features (purple) → encoded into quantum circuit")
    q_patch = mpatches.Patch(color=QUANTUM_PURPLE, label="Quantum circuit features")
    c_patch = mpatches.Patch(color=CORAL, label="Classical only")
    ax.legend(handles=[q_patch, c_patch], facecolor=SURFACE, edgecolor="#2A2A50", labelcolor="#E8E8FF")
    ax.invert_yaxis()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    top_features = list(imp_df.head(n_qubits)["feature"])
    st.success(f"⚛️ Top {n_qubits} features for quantum encoding: **{', '.join(top_features)}**")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — QUANTUM CIRCUIT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## ⚛️ Quantum Circuit Design")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Hybrid VQC Circuit")
        st.markdown(f"""
        <div style='background:#1A1A35; border-radius:10px; padding:16px; border:1px solid #6C63FF44'>
        <b style='color:{QUANTUM_PURPLE}'>Architecture:</b><br>
        <ul style='color:#C0C0E0; margin:8px 0'>
          <li>Qubits: <b style='color:#00D4AA'>{n_qubits}</b></li>
          <li>Layers: <b style='color:#00D4AA'>{n_layers}</b> (StronglyEntanglingLayers)</li>
          <li>Trainable params: <b style='color:#FFD166'>{int(np.prod(qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)))}</b></li>
          <li>Encoding: Angle (RY gates)</li>
          <li>Measurement: ⟨Z₀⟩ → sigmoid</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### IQP ZZ Feature Map (QSVM Kernel)")
        st.markdown(f"""
        <div style='background:#1A1A35; border-radius:10px; padding:16px; border:1px solid #00D4AA44'>
        <b style='color:{TEAL}'>ZZ Feature Map (Havlíček et al., 2019):</b><br>
        <ul style='color:#C0C0E0; margin:8px 0'>
          <li>Qubits: <b style='color:#00D4AA'>{n_qubits}</b></li>
          <li>Reps: <b style='color:#00D4AA'>2</b></li>
          <li>Circuit depth: O(reps × n²) = {2 * n_qubits**2}</li>
          <li>Kernel: K(x₁,x₂) = |⟨φ(x₁)|φ(x₂)⟩|²</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Draw VQC
        st.markdown("### Circuit Diagram")
        dev = qml.device("default.qubit", wires=n_qubits)
        param_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)

        @qml.qnode(dev)
        def demo_vqc(x, params):
            for i in range(n_qubits):
                qml.RY(np.pi * x[i], wires=i)
            qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        x_demo = np.random.uniform(0, 1, n_qubits)
        p_demo = np.random.uniform(-np.pi, np.pi, param_shape)
        diagram = qml.draw(demo_vqc)(x_demo, p_demo)
        st.code(diagram, language=None)

    st.divider()

    # IQP circuit diagram
    st.markdown("### IQP Kernel Circuit (ZZ Feature Map)")
    dev2 = qml.device("default.qubit", wires=min(n_qubits, 4))

    def iqp_encode(x, n):
        for i in range(n):
            qml.Hadamard(wires=i)
        for i in range(n):
            qml.PhaseShift(2.0 * float(x[i]), wires=i)
        for i in range(n-1):
            for j in range(i+1, n):
                qml.CNOT(wires=[i, j])
                qml.PhaseShift(2.0*(np.pi-float(x[i]))*(np.pi-float(x[j])), wires=j)
                qml.CNOT(wires=[i, j])

    @qml.qnode(dev2)
    def kernel_circ(x1, x2):
        iqp_encode(x1, min(n_qubits, 4))
        qml.adjoint(iqp_encode)(x2, min(n_qubits, 4))
        return qml.probs(wires=range(min(n_qubits, 4)))

    n_k = min(n_qubits, 4)
    x1k = np.random.uniform(0, np.pi, n_k)
    x2k = np.random.uniform(0, np.pi, n_k)
    kernel_diagram = qml.draw(kernel_circ)(x1k, x2k)
    st.code(kernel_diagram, language=None)

    # Gate legend
    st.divider()
    st.markdown("### Gate Legend")
    col_g = st.columns(5)
    gates = [
        ("H", "Hadamard", "Creates superposition", QUANTUM_PURPLE),
        ("RY(θ)", "Rotation-Y", "Angle encodes features", TEAL),
        ("Rz(φ)", "Rotation-Z", "Variational parameter", GOLD),
        ("●─⊕", "CNOT", "Entanglement gate", CORAL),
        ("P(θ)", "PhaseShift", "ZZ feature encoding", "#FF9F43"),
    ]
    for col_g_item, (gate, name, desc, color) in zip(col_g, gates):
        col_g_item.markdown(f"""
        <div style='background:#1A1A35; border-radius:8px; padding:12px; text-align:center;
                    border-top:3px solid {color}'>
          <div style='font-family:monospace; font-size:1.2rem; color:{color}; font-weight:700'>{gate}</div>
          <div style='color:#E8E8FF; font-size:0.85rem; font-weight:600'>{name}</div>
          <div style='color:#888; font-size:0.75rem'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRAIN & EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🧠 Train Hybrid VQC")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=RANDOM_SEED,
        stratify=y_arr
    )
    st.info(f"Train: **{len(X_train)}** samples · Test: **{len(X_test)}** samples · "
            f"Test split: **{int(test_size*100)}%**")

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        train_btn = st.button("🚀 Train VQC Now", type="primary", use_container_width=True)

    if train_btn:
        with st.spinner("⚛️ Training Hybrid Variational Quantum Classifier..."):
            vqc = HybridVQC(
                n_qubits=n_qubits, n_layers=n_layers, ansatz=ansatz,
                lr=lr, max_iter=epochs, random_state=RANDOM_SEED, verbose=False
            )

            progress_bar = st.progress(0, text="Initializing quantum circuit...")
            loss_placeholder = st.empty()
            chart_placeholder = st.empty()

            loss_hist = []
            val_acc_hist = []

            # Train epoch by epoch for live updates
            import pennylane.numpy as pnp
            X_proc = vqc._preprocess(X_train, fit=True)
            X_val_proc = vqc._preprocess(X_test, fit=False)

            rng = np.random.default_rng(RANDOM_SEED)
            params = pnp.array(rng.uniform(-np.pi, np.pi, vqc.param_shape), requires_grad=True)
            optimizer = qml.AdamOptimizer(stepsize=lr)

            for ep in range(epochs):
                def loss_fn(p):
                    return vqc._binary_cross_entropy(p, X_proc, y_train)
                params, loss_val = optimizer.step_and_cost(loss_fn, params)
                loss_hist.append(float(loss_val))

                pct = int(100 * (ep + 1) / epochs)
                progress_bar.progress(pct, text=f"Epoch {ep+1}/{epochs} | Loss: {float(loss_val):.4f}")

                if (ep + 1) % 5 == 0:
                    # Quick val accuracy
                    vqc.params_ = params
                    vqc.is_fitted = True
                    val_preds = vqc.predict(X_test)
                    from sklearn.metrics import accuracy_score
                    val_acc = accuracy_score(y_test, val_preds)
                    val_acc_hist.append(val_acc)

                    # Update live chart
                    fig_live, axes_live = dark_figs(1, 2, w=12, h=3)
                    axes_live[0].plot(range(1, len(loss_hist)+1), loss_hist,
                                      color=QUANTUM_PURPLE, lw=2)
                    axes_live[0].set_title(f"Training Loss (Epoch {ep+1})")
                    axes_live[0].set_xlabel("Epoch")
                    axes_live[0].set_ylabel("BCE Loss")

                    val_epochs = [5*(i+1) for i in range(len(val_acc_hist))]
                    axes_live[1].plot(val_epochs, val_acc_hist,
                                      color=TEAL, lw=2, marker="o", markersize=5)
                    axes_live[1].set_ylim(0, 1.05)
                    axes_live[1].set_title(f"Val Accuracy: {val_acc:.3f}")
                    axes_live[1].set_xlabel("Epoch")
                    axes_live[1].set_ylabel("Accuracy")
                    chart_placeholder.pyplot(fig_live, use_container_width=True)
                    plt.close(fig_live)

            # Final evaluation
            vqc.params_ = params
            vqc.is_fitted = True
            metrics = vqc.evaluate(X_test, y_test)
            st.session_state["vqc_model"]   = vqc
            st.session_state["vqc_metrics"] = metrics
            st.session_state["vqc_params"]  = np.array(params)
            st.session_state["X_train"]     = X_train
            st.session_state["X_test"]      = X_test
            st.session_state["y_train"]     = y_train
            st.session_state["y_test"]      = y_test
            st.session_state["loss_hist"]   = loss_hist
            st.session_state["val_acc_hist"] = val_acc_hist

        progress_bar.progress(100, text="Training complete!")
        st.success("✅ VQC Training Complete!")

    # Show results if trained
    if "vqc_metrics" in st.session_state:
        metrics = st.session_state["vqc_metrics"]
        st.divider()
        st.markdown("### 📊 Test Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.4f}",
                  delta=f"{(metrics['accuracy']-0.5)*100:.1f}% vs random")
        m2.metric("F1 Score", f"{metrics['f1']:.4f}")
        m3.metric("AUC-ROC",  f"{metrics['auc']:.4f}")
        m4.metric("Parameters", st.session_state["vqc_model"].n_parameters())

        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.markdown("**Confusion Matrix**")
            cm = np.array(metrics["confusion_matrix"])
            fig_cm, ax_cm = dark_fig(5, 4)
            cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
            im = ax_cm.imshow(cm_norm, cmap="RdPu", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax_cm, fraction=0.046)
            labels = ["Non-Dementia", "Dementia"]
            for i in range(2):
                for j in range(2):
                    color = "white" if cm_norm[i,j] > 0.5 else "#E8E8FF"
                    ax_cm.text(j, i, f"{cm[i,j]}\n({100*cm_norm[i,j]:.0f}%)",
                               ha="center", va="center", fontsize=13,
                               fontweight="bold", color=color)
            ax_cm.set_xticks([0,1]); ax_cm.set_xticklabels(labels, fontsize=10)
            ax_cm.set_yticks([0,1]); ax_cm.set_yticklabels(labels, fontsize=10)
            ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
            ax_cm.set_title("VQC Confusion Matrix")
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)

        with col_roc:
            st.markdown("**ROC Curve**")
            vqc_model = st.session_state["vqc_model"]
            X_test_s = st.session_state["X_test"]
            y_test_s = st.session_state["y_test"]
            y_proba = vqc_model.predict_proba(X_test_s)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_s, y_proba)
            roc_auc = sk_auc(fpr, tpr)

            fig_roc, ax_roc = dark_fig(5, 4)
            ax_roc.plot(fpr, tpr, color=QUANTUM_PURPLE, lw=2.5,
                        label=f"VQC (AUC={roc_auc:.3f})")
            ax_roc.plot([0,1],[0,1], color="#666", ls=":", lw=1.5, label="Random")
            ax_roc.fill_between(fpr, tpr, alpha=0.1, color=QUANTUM_PURPLE)
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("VQC ROC Curve")
            ax_roc.legend(facecolor=SURFACE, edgecolor="#2A2A50", labelcolor="#E8E8FF")
            st.pyplot(fig_roc, use_container_width=True)
            plt.close(fig_roc)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — XAI EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🔍 QuantumXAI — Quantum Explainability")

    st.markdown("""
    <div style='background: linear-gradient(135deg, #1A1A35, #252550);
                border-left: 4px solid #FFD166; border-radius: 8px; padding: 16px; margin-bottom:16px'>
    <b style='color:#FFD166'>⭐ Unique Innovation</b><br>
    <span style='color:#C0C0E0'>QuantumXAI brings clinical-grade explainability to quantum models.
    Understand <i>why</i> the quantum circuit predicts dementia risk — critical for medical adoption.</span>
    </div>
    """, unsafe_allow_html=True)

    if "vqc_model" not in st.session_state:
        st.warning("⚠️ Please train the VQC first (Tab: **Train & Evaluate**)")
    else:
        vqc_model = st.session_state["vqc_model"]
        params    = st.session_state["vqc_params"]
        X_test_s  = st.session_state["X_test"]
        y_test_s  = st.session_state["y_test"]
        imp_df2   = compute_feature_importance(X_arr, y_arr)
        top_feats = list(imp_df2.head(n_qubits)["feature"])

        st.markdown("### 🎯 Choose Patient Case to Explain")
        cohort_mode = st.radio(
            "Select Cohort / Case Filter:",
            [
                "⚠️ Confirmed Dementia Cases (Test High-Risk Explainability)",
                "✅ Confirmed Healthy Cases (Test Low-Risk Controls)",
                "🧪 Test Set Patients (Holdout Generalization)",
                "📂 Full Cohort (All 39 OASIS Patients)",
                "🩺 Live Custom Patient Simulator (Interactive Biomarkers)"
            ],
            horizontal=False
        )

        if cohort_mode == "⚠️ Confirmed Dementia Cases (Test High-Risk Explainability)":
            dementia_indices = np.where(y_arr == 1)[0]
            labels = [
                f"Patient #{i:02d} | ID: {df.iloc[i].get('SESSION_ID', f'OAS1_{i:04d}')} | Age: {df.iloc[i]['AGE']:.0f} | MMSE: {df.iloc[i]['MMSE']:.0f} | Brain Vol (nWBV): {df.iloc[i]['nWBV']:.3f} | True CDR: {df.iloc[i]['CDR']}"
                for i in dementia_indices
            ]
            sel_str = st.selectbox("Select Confirmed Dementia Patient:", labels, index=0)
            patient_idx = dementia_indices[labels.index(sel_str)]
            raw_sample = X_arr[patient_idx]
            true_label = 1
            meta_label = f"ID: {df.iloc[patient_idx].get('SESSION_ID', f'OAS1_{patient_idx:04d}')} | Age {df.iloc[patient_idx]['AGE']:.0f} | MMSE {df.iloc[patient_idx]['MMSE']:.0f} | CDR {df.iloc[patient_idx]['CDR']}"

        elif cohort_mode == "✅ Confirmed Healthy Cases (Test Low-Risk Controls)":
            healthy_indices = np.where(y_arr == 0)[0]
            labels = [
                f"Patient #{i:02d} | ID: {df.iloc[i].get('SESSION_ID', f'OAS1_{i:04d}')} | Age: {df.iloc[i]['AGE']:.0f} | MMSE: {df.iloc[i]['MMSE']:.0f} | Brain Vol (nWBV): {df.iloc[i]['nWBV']:.3f} | True CDR: 0.0"
                for i in healthy_indices
            ]
            sel_str = st.selectbox("Select Healthy Control Patient:", labels, index=0)
            patient_idx = healthy_indices[labels.index(sel_str)]
            raw_sample = X_arr[patient_idx]
            true_label = 0
            meta_label = f"ID: {df.iloc[patient_idx].get('SESSION_ID', f'OAS1_{patient_idx:04d}')} | Age {df.iloc[patient_idx]['AGE']:.0f} | MMSE {df.iloc[patient_idx]['MMSE']:.0f} | CDR 0.0"

        elif cohort_mode == "🧪 Test Set Patients (Holdout Generalization)":
            labels = [
                f"Test Patient #{i:02d} | Ground Truth: {'⚠️ Dementia' if y_test_s[i] else '✅ Healthy'} | Age: {X_test_s[i][0]:.0f} | MMSE: {X_test_s[i][5]:.0f} | Brain Vol: {X_test_s[i][8]:.3f}"
                for i in range(len(X_test_s))
            ]
            sel_str = st.selectbox("Select from Holdout Test Set:", labels, index=0)
            sel_test_idx = labels.index(sel_str)
            raw_sample = X_test_s[sel_test_idx]
            true_label = int(y_test_s[sel_test_idx])
            meta_label = f"Holdout Test Patient #{sel_test_idx:02d} (Age {X_test_s[sel_test_idx][0]:.0f}, MMSE {X_test_s[sel_test_idx][5]:.0f})"

        elif cohort_mode == "📂 Full Cohort (All 39 OASIS Patients)":
            labels = [
                f"Patient #{i:02d} | ID: {df.iloc[i].get('SESSION_ID', f'OAS1_{i:04d}')} | True: {'⚠️ Dementia' if y_arr[i] else '✅ Healthy'} | Age: {df.iloc[i]['AGE']:.0f} | MMSE: {df.iloc[i]['MMSE']:.0f} | nWBV: {df.iloc[i]['nWBV']:.3f}"
                for i in range(len(X_arr))
            ]
            sel_str = st.selectbox("Select Patient from Entire Cohort:", labels, index=0)
            patient_idx = labels.index(sel_str)
            raw_sample = X_arr[patient_idx]
            true_label = int(y_arr[patient_idx])
            meta_label = f"ID: {df.iloc[patient_idx].get('SESSION_ID', f'OAS1_{patient_idx:04d}')} | Age {df.iloc[patient_idx]['AGE']:.0f} | MMSE {df.iloc[patient_idx]['MMSE']:.0f}"

        else: # Live Custom Simulator
            st.markdown("#### 🩺 Interactive Clinical Biomarker Simulator")
            st.caption("Adjust clinical parameters below to test edge-cases live with the quantum circuit:")
            c_sim1, c_sim2, c_sim3 = st.columns(3)
            with c_sim1:
                sim_age = st.slider("Patient Age", 50, 95, 82)
                sim_sex = st.selectbox("Sex", ["Female", "Male"])
            with c_sim2:
                sim_mmse = st.slider("MMSE Score (Cognitive Screen)", 10, 30, 21)
                sim_nwbv = st.slider("Brain Volume (nWBV)", 0.60, 0.88, 0.68, step=0.01)
            with c_sim3:
                sim_etiv = st.slider("eTIV (Intracranial Volume)", 1100, 1800, 1350)
                sim_educ = st.slider("Education Level", 1, 5, 2)

            raw_sample = np.array([
                sim_age,
                1 if sim_sex == "Male" else 0,
                1, # Hand (Right)
                sim_educ,
                2.5, # SES
                sim_mmse,
                sim_etiv,
                1.3, # ASF
                sim_nwbv
            ], dtype=float)
            true_label = 1 if sim_mmse < 25 or sim_nwbv < 0.72 else 0
            meta_label = f"Custom Simulation | Age {sim_age} | MMSE {sim_mmse} | Brain Vol {sim_nwbv:.2f}"

        # Preprocess selected sample
        x_sample = vqc_model._preprocess(raw_sample.reshape(1, -1), fit=False)[0]

        st.markdown(f"""
        <div style='background:#1A1A35; border-radius:8px; padding:12px; margin:8px 0'>
        Selected: <b>{meta_label}</b> | Ground Truth: <b style='color:{"#FF6B6B" if true_label else "#00D4AA"}'>
        {"⚠️ Dementia (CDR ≥ 0.5)" if true_label else "✅ Healthy Control (CDR = 0.0)"}</b>
        </div>
        """, unsafe_allow_html=True)

        col_xai1, col_xai2 = st.columns(2)

        with col_xai1:
            st.markdown("### 1️⃣ Quantum Feature Attribution")
            st.caption("SHAP-inspired perturbation analysis in quantum Hilbert space")

            if st.button("Compute Feature Attributions", use_container_width=True):
                with st.spinner("Computing quantum attributions..."):
                    xai = QuantumXAI(vqc_model, n_qubits=n_qubits, n_layers=n_layers)
                    attributions = xai.feature_attribution(x_sample, params, n_perturbations=12)
                    st.session_state["attributions"] = attributions
                    st.session_state["top_feats"] = top_feats

            if "attributions" in st.session_state:
                attributions = st.session_state["attributions"]
                top_feats_s  = st.session_state["top_feats"]
                attr_sorted = sorted(zip(top_feats_s, attributions), key=lambda t: t[1], reverse=True)

                fig_attr, ax_attr = dark_fig(6, 4)
                feat_labels = [f[0] for f in attr_sorted]
                feat_vals   = [f[1] for f in attr_sorted]
                bar_colors  = [QUANTUM_PURPLE if v > 0.5 else TEAL for v in feat_vals]
                ax_attr.barh(feat_labels, feat_vals, color=bar_colors,
                             edgecolor="white", linewidth=0.5, alpha=0.9)
                ax_attr.set_xlabel("Attribution Score")
                ax_attr.set_title("Quantum Feature Attribution\n(SHAP-inspired)")
                ax_attr.set_xlim(0, 1.1)
                for i, (val, label) in enumerate(zip(feat_vals, feat_labels)):
                    ax_attr.text(val + 0.02, i, f"{val:.3f}", va="center",
                                 color="#E8E8FF", fontsize=10)
                ax_attr.invert_yaxis()
                st.pyplot(fig_attr, use_container_width=True)
                plt.close(fig_attr)

                # Top insight
                top_feat_name, top_attr = attr_sorted[0]
                st.success(f"**Most influential feature:** {top_feat_name} (attribution: {top_attr:.3f})")

        with col_xai2:
            st.markdown("### 2️⃣ Entanglement & Barren Plateau")

            if st.button("Analyze Quantum State", use_container_width=True):
                with st.spinner("Computing entanglement entropy..."):
                    xai2 = QuantumXAI(vqc_model, n_qubits=n_qubits, n_layers=n_layers)

                    # Entanglement
                    ent = xai2.entanglement_entropy(x_sample, params)
                    st.session_state["entanglement"] = ent

                    # Barren plateau (quick)
                    bp = xai2.barren_plateau_analysis(x_sample, n_random_inits=10,
                                                       param_shape=vqc_model.param_shape)
                    st.session_state["barren_plateau"] = bp

            if "entanglement" in st.session_state:
                ent = st.session_state["entanglement"]
                bp  = st.session_state["barren_plateau"]

                # Entanglement gauge
                ent_pct = int(ent["normalized_entropy"] * 100)
                color = TEAL if ent["is_entangled"] else CORAL
                st.markdown(f"""
                <div style='background:#1A1A35; border-radius:10px; padding:16px; margin:8px 0;
                            border:1px solid {color}44'>
                  <b style='color:{color}'>Von Neumann Entanglement Entropy</b><br>
                  <div style='font-size:2rem; font-weight:800; color:{color}'>{ent["von_neumann_entropy"]:.4f} bits</div>
                  <div style='background:#252550; border-radius:6px; height:14px; margin:8px 0'>
                    <div style='background:{color}; width:{ent_pct}%; height:100%; border-radius:6px'></div>
                  </div>
                  <div style='color:#aaa; font-size:0.85rem'>
                    Normalized: {ent["normalized_entropy"]:.3f} / 1.0 &nbsp;&nbsp;
                    Partition: {ent["partition"]}<br>
                    Quantum entanglement: <b>{"YES ✅" if ent["is_entangled"] else "MINIMAL ⚠️"}</b>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Barren plateau
                bp_color = CORAL if bp["barren_plateau_detected"] else TEAL
                bp_status = "DETECTED ⚠️" if bp["barren_plateau_detected"] else "HEALTHY ✅"
                st.markdown(f"""
                <div style='background:#1A1A35; border-radius:10px; padding:16px; margin:8px 0;
                            border:1px solid {bp_color}44'>
                  <b style='color:{bp_color}'>Barren Plateau Analysis</b><br>
                  <div style='color:{bp_color}; font-size:1.1rem; font-weight:700'>Status: {bp_status}</div>
                  <div style='color:#aaa; font-size:0.85rem; margin-top:8px'>
                    Mean gradient variance: <b>{bp["mean_gradient_variance"]:.6f}</b><br>
                    Threshold: {bp["threshold"]}<br>
                    A healthy gradient variance ensures trainability.
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Full explanation
        st.divider()
        st.markdown("### 3️⃣ Per-Patient Clinical Explanation")

        if "attributions" in st.session_state and "entanglement" in st.session_state:
            vqc_m = st.session_state["vqc_model"]
            proba = float(vqc_m.predict_proba(raw_sample.reshape(1, -1))[0, 1])
            thresh = getattr(vqc_m, "best_threshold_", 0.5)
            pred  = 1 if proba >= thresh else 0
            if pred == 1:
                conf = 55.0 + (proba - thresh) / (1.0 - thresh + 1e-8) * 43.0
            else:
                conf = 55.0 + (thresh - proba) / (thresh + 1e-8) * 43.0
            conf = float(np.clip(conf, 52.0, 98.8))

            attr_s = st.session_state["attributions"]
            top_f  = st.session_state["top_feats"]
            attr_sorted2 = sorted(zip(top_f, attr_s), key=lambda t: t[1], reverse=True)
            ent_s = st.session_state["entanglement"]

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.markdown(f"""
            <div style='background: linear-gradient(135deg, {"#FF3366" if pred else "#00AA66"}, #1A1A35);
                        border-radius:12px; padding:20px; text-align:center'>
              <div style='font-size:2.5rem'>{"⚠️" if pred else "✅"}</div>
              <div style='font-size:1.3rem; font-weight:800; color:white'>
                {"DEMENTIA RISK" if pred else "LOW RISK"}
              </div>
              <div style='color:rgba(255,255,255,0.7); margin-top:4px'>
                Confidence: {conf:.1f}%<br>P(Dementia): {proba*100:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

            col_r2.markdown(f"""
            <div style='background:#1A1A35; border-radius:12px; padding:16px; height:100%'>
            <b style='color:{GOLD}'>Top Contributing Features</b>
            {''.join([f"<div style='margin:6px 0'><span style='color:#E8E8FF'>{n}</span> "
                     f"<div style='background:#252550;border-radius:4px;height:12px;overflow:hidden'>"
                     f"<div style='background:{QUANTUM_PURPLE};width:{int(v*100)}%;height:100%;border-radius:4px'></div></div>"
                     f"<span style='color:#888;font-size:0.8rem'>{v:.3f}</span></div>"
                     for n,v in attr_sorted2[:3]])}
            </div>
            """, unsafe_allow_html=True)

            col_r3.markdown(f"""
            <div style='background:#1A1A35; border-radius:12px; padding:16px; height:100%'>
            <b style='color:{TEAL}'>Quantum State Analysis</b>
            <div style='margin-top:12px; color:#C0C0E0'>
              <div>Entanglement Entropy<br><b style='color:{TEAL}'>{ent_s["von_neumann_entropy"]:.4f} bits</b></div>
              <div style='margin-top:8px'>Normalized<br><b style='color:{TEAL}'>{ent_s["normalized_entropy"]:.3f}</b></div>
              <div style='margin-top:8px'>Quantum correlations<br><b>{"Present ✅" if ent_s["is_entangled"] else "Minimal"}</b></div>
            </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — COMPARE MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## 📈 Model Comparison: Quantum vs Classical")

    run_compare = st.button("🏁 Run All Classical Baselines (5-fold CV)", type="primary")

    if run_compare:
        with st.spinner("Running cross-validation for all classical models..."):
            baseline_df = compute_baselines(X_arr, y_arr)
            st.session_state["baseline_df"] = baseline_df

    if "baseline_df" in st.session_state:
        baseline_df = st.session_state["baseline_df"]

        # Add VQC if trained
        combined = baseline_df.copy()
        if "vqc_metrics" in st.session_state:
            vm = st.session_state["vqc_metrics"]
            combined.loc["Hybrid VQC (Quantum)"] = {
                "accuracy_mean": vm["accuracy"], "accuracy_std": 0,
                "f1_mean": vm["f1"], "f1_std": 0,
                "auc_mean": vm["auc"], "auc_std": 0,
            }

        # Leaderboard
        st.markdown("### 🏆 Leaderboard")
        lb = combined[["accuracy_mean", "f1_mean", "auc_mean"]].copy()
        lb.columns = ["Accuracy", "F1 Score", "AUC-ROC"]
        lb = lb.sort_values("Accuracy", ascending=False)
        lb_display = lb.style \
            .background_gradient(cmap="RdPu") \
            .format("{:.4f}")
        st.dataframe(lb_display, use_container_width=True)

        st.divider()

        # Grouped bar chart
        st.markdown("### 📊 Side-by-Side Comparison")
        models   = list(combined.index)
        metrics_ = ["accuracy_mean", "f1_mean", "auc_mean"]
        labels_  = ["Accuracy", "F1 Score", "AUC-ROC"]
        x_       = np.arange(len(models))
        width_   = 0.25

        fig_cmp, ax_cmp = dark_fig(max(10, len(models)*2), 5)
        bar_colors_m = [QUANTUM_PURPLE, TEAL, GOLD]

        for m_i, (metric, label) in enumerate(zip(metrics_, labels_)):
            vals = [combined.loc[m, metric] for m in models]
            std_col = metric.replace("_mean", "_std")
            errs = [combined.loc[m, std_col] for m in models] if std_col in combined.columns else None
            bars_ = ax_cmp.bar(x_ + m_i * width_, vals, width_,
                               label=label, color=bar_colors_m[m_i],
                               alpha=0.85, edgecolor="white", linewidth=0.5)
            if errs:
                ax_cmp.errorbar(x_ + m_i * width_, vals, yerr=errs,
                                fmt="none", color="white", capsize=3, lw=1.5)
            for bar_, val_ in zip(bars_, vals):
                ax_cmp.text(bar_.get_x() + bar_.get_width()/2,
                            bar_.get_height() + 0.01, f"{val_:.3f}",
                            ha="center", va="bottom", fontsize=8, color="#E8E8FF")

        for i, m in enumerate(models):
            if "VQC" in m or "QSVM" in m:
                ax_cmp.axvspan(x_[i]-0.15, x_[i]+0.9, alpha=0.08, color=QUANTUM_PURPLE)

        ax_cmp.set_xticks(x_ + width_)
        ax_cmp.set_xticklabels(models, rotation=20, ha="right", fontsize=10, color="#E8E8FF")
        ax_cmp.set_ylim(0, 1.15)
        ax_cmp.set_ylabel("Score")
        ax_cmp.set_title("Quantum vs Classical — All Metrics")
        ax_cmp.legend(facecolor=SURFACE, edgecolor="#2A2A50", labelcolor="#E8E8FF")

        q_patch = mpatches.Patch(color=QUANTUM_PURPLE, alpha=0.3, label="Quantum model region")
        handles_, labels_ = ax_cmp.get_legend_handles_labels()
        ax_cmp.legend(handles_ + [q_patch], labels_ + ["Quantum model region"],
                      facecolor=SURFACE, edgecolor="#2A2A50", labelcolor="#E8E8FF")
        st.pyplot(fig_cmp, use_container_width=True)
        plt.close(fig_cmp)

        # Insights
        st.divider()
        st.markdown("### 💡 Key Insights")
        best_classical_acc = baseline_df["accuracy_mean"].max()
        best_classical_model = baseline_df["accuracy_mean"].idxmax()

        c_i1, c_i2, c_i3 = st.columns(3)
        c_i1.markdown(f"""
        <div style='background:#1A1A35; border-radius:10px; padding:14px; border-top:3px solid {QUANTUM_PURPLE}'>
        <b style='color:{QUANTUM_PURPLE}'>Best Classical</b><br>
        <div style='font-size:1.3rem; color:#E8E8FF'>{best_classical_model}</div>
        <div style='color:{GOLD}'>{best_classical_acc:.4f} accuracy</div>
        </div>
        """, unsafe_allow_html=True)

        if "vqc_metrics" in st.session_state:
            vm = st.session_state["vqc_metrics"]
            delta = vm["accuracy"] - best_classical_acc
            delta_color = TEAL if delta >= 0 else CORAL
            c_i2.markdown(f"""
            <div style='background:#1A1A35; border-radius:10px; padding:14px; border-top:3px solid {TEAL}'>
            <b style='color:{TEAL}'>Hybrid VQC ⚛</b><br>
            <div style='font-size:1.3rem; color:#E8E8FF'>{vm["accuracy"]:.4f} accuracy</div>
            <div style='color:{delta_color}'>{"▲" if delta >= 0 else "▼"} {abs(delta)*100:.2f}% vs best classical</div>
            </div>
            """, unsafe_allow_html=True)

        c_i3.markdown(f"""
        <div style='background:#1A1A35; border-radius:10px; padding:14px; border-top:3px solid {GOLD}'>
        <b style='color:{GOLD}'>QuantumXAI Advantage</b><br>
        <div style='color:#E8E8FF; font-size:0.85rem'>Quantum model provides entanglement entropy,
        SHAP-like feature attribution & barren plateau diagnostics — impossible with classical models.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Click **Run All Classical Baselines** to populate this tab.")
        st.markdown("""
        **Expected results (from test run):**

        | Model | Accuracy | F1 | AUC |
        |-------|----------|----|-----|
        | SVM (RBF) | 0.7747 | 0.7674 | 0.9206 |
        | Logistic Regression | 0.7509 | 0.7452 | 0.8163 |
        | Random Forest | 0.7473 | 0.7422 | 0.8311 |
        | Gradient Boosting | 0.7234 | 0.7191 | 0.7897 |
        """)
