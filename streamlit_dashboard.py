"""
Continual Learning — Interactive Streamlit Dashboard
=====================================================
Reads `scalability_results.json` (fixed file, no upload option)

Run:  streamlit run streamlit_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Continual Learning Dashboard",
    page_icon="🧠",
    layout="wide",
)

METHODS = ["finetune", "ewc", "replay"]
LABELS  = {"finetune": "Fine-tuning", "ewc": "EWC", "replay": "Replay"}
COLORS  = {"finetune": "#378ADD", "ewc": "#1D9E75", "replay": "#BA7517"}

# ── Load data (FIXED FILE ONLY) ──────────────────────────────
RESULTS_PATH = "scalability_results.json"

@st.cache_data
def load_results(path: str):
    with open(path) as f:
        return json.load(f)

if not os.path.exists(RESULTS_PATH):
    st.error("❌ scalability_results.json not found. Place it in the same folder.")
    st.stop()

data = load_results(RESULTS_PATH)

# ── Helper metrics ───────────────────────────────────────────
def avg_acc_curve(mat):
    m = np.array(mat)
    return [float(np.mean(m[t][:t+1])) for t in range(m.shape[0])]

def forgetting_curve(mat):
    m = np.array(mat)
    n = m.shape[0]
    c = [0.0]
    for t in range(1, n):
        c.append(float(np.mean([max(0, m[i][i] - m[t][i]) for i in range(t)])))
    return c

# ── Header ───────────────────────────────────────────────────
st.title("🧠 Continual Learning — Scalability Dashboard")

datasets_available = list(data.keys())

# ── Sidebar controls ─────────────────────────────────────────
st.sidebar.header("⚙️ Controls")

selected_methods = st.sidebar.multiselect(
    "Methods", METHODS, default=METHODS,
    format_func=lambda m: LABELS[m]
)

selected_ds = st.sidebar.selectbox(
    "Dataset", datasets_available,
    format_func=lambda d: d.upper()
)

# ── Summary cards ────────────────────────────────────────────
st.header(f"📊 Summary — {selected_ds.upper()}")

cols = st.columns(len(selected_methods))
for i, mth in enumerate(selected_methods):
    r = data[selected_ds][mth]
    with cols[i]:
        st.metric(f"{LABELS[mth]} — Final Accuracy", f"{r['final_acc']:.1f}%")
        st.metric("Forgetting", f"{r['forgetting']:.1f}%")
        st.metric("Total Time", f"{sum(r['times_s']):.0f}s")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────
tab_acc, tab_fgt, tab_mem, tab_heat, tab_deg, tab_raw = st.tabs([
    "📈 Accuracy", "📉 Forgetting", "💾 Memory",
    "🗺️ Heatmaps", "⚖️ Degradation", "📋 Raw Data"
])

ds_data = data[selected_ds]
T = len(ds_data[METHODS[0]]["acc_matrix"])
xs = list(range(1, T + 1))

# ── Accuracy ─────────────────────────────────────────────────
with tab_acc:
    fig = go.Figure()
    for mth in selected_methods:
        curve = avg_acc_curve(ds_data[mth]["acc_matrix"])
        fig.add_trace(go.Scatter(
            x=xs, y=curve, mode="lines+markers",
            name=LABELS[mth],
            line=dict(color=COLORS[mth], width=3),
        ))
    fig.update_layout(
        title=f"Average Accuracy — {selected_ds.upper()}",
        xaxis_title="Tasks",
        yaxis_title="Accuracy (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Forgetting ───────────────────────────────────────────────
with tab_fgt:
    fig = go.Figure()
    for mth in selected_methods:
        curve = forgetting_curve(ds_data[mth]["acc_matrix"])
        fig.add_trace(go.Scatter(
            x=xs, y=curve, mode="lines+markers",
            name=LABELS[mth],
            line=dict(color=COLORS[mth], width=3),
        ))
    fig.update_layout(
        title=f"Forgetting — {selected_ds.upper()}",
        xaxis_title="Tasks",
        yaxis_title="Forgetting (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Memory ───────────────────────────────────────────────────
with tab_mem:
    fig = go.Figure()
    for mth in selected_methods:
        fig.add_trace(go.Bar(
            x=[f"T{t+1}" for t in range(T)],
            y=ds_data[mth]["mem_mb"],
            name=LABELS[mth],
            marker_color=COLORS[mth]
        ))
    fig.update_layout(
        title="Memory Usage",
        yaxis_title="MB",
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Heatmaps ─────────────────────────────────────────────────
with tab_heat:
    cols = st.columns(len(selected_methods))
    for i, mth in enumerate(selected_methods):
        mat = np.array(ds_data[mth]["acc_matrix"])
        with cols[i]:
            fig = px.imshow(mat, text_auto=".1f",
                            color_continuous_scale="RdYlGn",
                            zmin=0, zmax=100,
                            title=LABELS[mth])
            st.plotly_chart(fig, use_container_width=True)

# ── Degradation ──────────────────────────────────────────────
with tab_deg:
    if "cifar10" in data and "cifar100" in data:
        for mth in selected_methods:
            a10 = data["cifar10"][mth]["final_acc"]
            a100 = data["cifar100"][mth]["final_acc"]
            deg = 100 * (a10 - a100) / a10
            st.write(f"{LABELS[mth]} → Degradation: {deg:.1f}%")

# ── Raw Data ─────────────────────────────────────────────────
with tab_raw:
    st.json(data)

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("Built for continual learning project 🚀")