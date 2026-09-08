import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

from src.data.feature_extraction import FeatureExtractor
from src.data.windowing import WindowBuilder
from src.models.world_model import build_world_model
from src.prediction.engine import PredictionEngine
from src.explainability.shap_explainer import SHAPExplainer
from src.data.dataset import STAGE_NAMES

st.set_page_config(
    page_title="SIH-26153: AI Network Attack Forecaster",
    page_icon="🛡️",
    layout="wide",
)

STAGE_COLORS = {
    "Benign": "#2ecc71",
    "Reconnaissance": "#f39c12",
    "Initial Access": "#e74c3c",
    "Lateral Movement": "#9b59b6",
    "Command and Control": "#e67e22",
    "Exfiltration": "#c0392b",
}


@st.cache_resource
def load_model(checkpoint_path: str, model_type: str, input_size: int, config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_world_model(
        model_type=model_type,
        input_size=input_size,
        num_classes=6,
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, device


def plot_timeline(timeline: list[dict]):
    stages = [entry["predicted_stage"] for entry in timeline]
    confidences = [entry["confidence"] for entry in timeline]
    windows = [entry["window_index"] for entry in timeline]
    colors = [STAGE_COLORS.get(s, "#95a5a6") for s in stages]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=windows, y=confidences,
        mode="lines+markers",
        marker=dict(color=colors, size=10),
        line=dict(color="#3498db", width=2),
        text=stages,
        hovertemplate="Window %{x}<br>Stage: %{text}<br>Confidence: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Attack Stage Timeline",
        xaxis_title="Window Index",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=400,
    )
    return fig


def plot_stage_distribution(timeline: list[dict]):
    stages = [entry["predicted_stage"] for entry in timeline]
    stage_counts = pd.Series(stages).value_counts()
    colors = [STAGE_COLORS.get(s, "#95a5a6") for s in stage_counts.index]

    fig = go.Figure(data=[go.Pie(
        labels=stage_counts.index,
        values=stage_counts.values,
        marker=dict(colors=colors),
        hole=0.3,
    )])
    fig.update_layout(title="Stage Distribution", template="plotly_white", height=400)
    return fig


def plot_forecast(forecast: list[dict]):
    steps = [f["step"] for f in forecast]
    stage_names = [f["predicted_stage"] for f in forecast]
    confs = [f["confidence"] for f in forecast]
    colors = [STAGE_COLORS.get(s, "#95a5a6") for s in stage_names]

    fig = go.Figure(data=[go.Bar(
        x=steps, y=confs,
        marker_color=colors,
        text=stage_names,
        textposition="auto",
    )])
    fig.update_layout(
        title="K-Step Forecast",
        xaxis_title="Forecast Step",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=350,
    )
    return fig


def plot_feature_importance(features: list[dict]):
    names = [f["feature"] for f in features]
    importances = [f["importance"] for f in features]

    fig = go.Figure(data=[go.Bar(
        y=names[::-1], x=importances[::-1],
        orientation="h",
        marker_color="#3498db",
    )])
    fig.update_layout(
        title="Top Driving Features (SHAP)",
        xaxis_title="Mean |SHAP value|",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_probability_heatmap(timeline: list[dict]):
    prob_matrix = []
    for entry in timeline[:100]:
        probs = entry["probabilities"]
        prob_matrix.append([probs.get(stage, 0) for stage in STAGE_NAMES])

    fig = go.Figure(data=go.Heatmap(
        z=list(zip(*prob_matrix)),
        x=[f"W{i}" for i in range(len(prob_matrix))],
        y=STAGE_NAMES,
        colorscale="YlOrRd",
    ))
    fig.update_layout(
        title="Stage Probability Heatmap Over Time",
        template="plotly_white",
        height=350,
    )
    return fig


def main():
    st.title("🛡️ SIH-26153: AI Network Attack Forecaster")
    st.markdown("**Predictive Cyber Defence using World Models**")

    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Upload & Analyze", "About", "Architecture"])

    with tab1:
        st.header("Live Dashboard")

        col1, col2 = st.columns([3, 1])
        with col2:
            model_type = st.selectbox("Model", ["lstm", "transformer", "gru"], index=0)
            k_steps = st.slider("Forecast Horizon (K)", 1, 10, 5)
            st.markdown("---")
            st.markdown("**Model Status**")
            if Path("checkpoints/best_model.pt").exists():
                st.success("Model loaded")
            else:
                st.warning("No trained model found. Run `train.py` first.")

        with col1:
            st.info("Upload a CSV or PCAP file in the 'Upload & Analyze' tab to see predictions here.")

    with tab2:
        st.header("Upload & Analyze Network Traffic")
        uploaded_file = st.file_uploader("Upload CSV (CIC-IDS format) or PCAP", type=["csv", "pcap"])

        if uploaded_file is not None:
            with st.spinner("Processing..."):
                save_path = Path(f"/tmp/{uploaded_file.name}")
                save_path.write_bytes(uploaded_file.read())

                extractor = FeatureExtractor()
                try:
                    df, feature_cols, label_col = extractor.process_file(save_path)
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    return

                st.success(f"Loaded {len(df)} flows with {len(feature_cols)} features")

                window_builder = WindowBuilder(window_size=60, slide_size=30)
                X, y, feature_names = window_builder.build_windows(df, feature_cols, label_col=label_col)

                if len(X) == 0:
                    st.warning("Not enough data to create windows. Try a larger file.")
                    return

                X_scaled, _ = extractor.normalize(X.reshape(-1, X.shape[-1]))
                X_scaled = X_scaled.reshape(X.shape)

                if Path("checkpoints/best_model.pt").exists():
                    model_input_size = 2 * len(feature_cols)
                    model, device = load_model(
                        "checkpoints/best_model.pt", model_type,
                        model_input_size, yaml.safe_load(open("configs/default.yaml"))
                    )
                    engine = PredictionEngine(model, device, sequence_length=10)
                    timeline = engine.generate_timeline(X_scaled, k_steps=k_steps)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plot_timeline(timeline), use_container_width=True)
                    with col2:
                        st.plotly_chart(plot_stage_distribution(timeline), use_container_width=True)

                    st.plotly_chart(plot_probability_heatmap(timeline), use_container_width=True)

                    st.subheader("K-Step Forecast (Latest Window)")
                    forecast = timeline[-1]["forecast"]
                    st.plotly_chart(plot_forecast(forecast), use_container_width=True)

                    st.subheader("Top Driving Features")
                    try:
                        explainer = SHAPExplainer(model, feature_names, device)
                        sequences, _ = engine._build_sequences(X_scaled)
                        explanation = explainer.explain_sequence(sequences[-1])
                        st.plotly_chart(plot_feature_importance(explanation["top_driving_features"]),
                                        use_container_width=True)
                    except Exception as e:
                        st.warning(f"Explainability unavailable: {e}")

                    with st.expander("Prediction Details"):
                        for entry in timeline[:20]:
                            st.write(f"**Window {entry['window_index']}:** {entry['predicted_stage']} "
                                     f"(confidence: {entry['confidence']:.3f})")
                else:
                    st.warning("No trained model available. Please run `python train.py` first.")

    with tab3:
        st.header("About This Project")
        st.markdown("""
        ### AI-based Network Attack Forecasting

        This system uses **World Models** to learn network behavior dynamics and predict
        attacker progression through MITRE ATT&CK stages.

        **Key Innovation:** Instead of binary classification (attack/benign), we predict
        the *trajectory* of an attack through stages:
        - Reconnaissance
        - Initial Access
        - Lateral Movement
        - Command & Control
        - Exfiltration

        **Technical Stack:**
        - LSTM / Transformer / GRU sequence models
        - SHAP-based explainability
        - Time-windowed feature engineering
        - Real-time prediction engine
        """)

    with tab4:
        st.header("System Architecture")
        st.markdown("""
        ```
        PCAP/CSV Input
              │
              ▼
        ┌─────────────────┐
        │ Feature Extraction│  ← Flow-level + Packet-level features
        │    Pipeline       │
        └────────┬─────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Window Builder   │  ← Time discretisation (60s windows)
        │                  │
        └────────┬─────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  World Model     │  ← LSTM / Transformer / GRU
        │  (P(S_{t+1}|S_t))│
        └────────┬─────────┘
                 │
        ┌────────┼────────┐
        │        │        │
        ▼        ▼        ▼
        Stage  K-Step   SHAP
        Pred.  Forecast Explain.
        """)


if __name__ == "__main__":
    import yaml
    main()
