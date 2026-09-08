# SIH-26153: AI-based Network Attack Forecasting
## Technical Presentation — 5 Slides

---

## SLIDE 1: Problem Statement & Motivation

### Title: Beyond Detection — Predicting Cyber Attacks Before They Happen

**The Problem:**
- Current IDS classify traffic as benign/malicious (binary, reactive)
- Attackers operate in stages over time — a single packet view misses the story
- SOC analysts face alert fatigue with millions of daily events

**Our Insight:**
> An infiltration is a *process unfolding over time*, not a single anomalous packet.

**What We Built:**
- A **World Model** that learns network state-transition dynamics: P(S_{t+1} | S_t)
- Forecasts the attacker's trajectory through MITRE ATT&CK stages
- Provides explainable predictions with SHAP feature attribution

**Impact:** Proactive defense — predict the next attack stage before it occurs.

---

## SLIDE 2: Technical Architecture

### Title: System Architecture — End-to-End Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  PCAP / CSV  │────▶│   Feature    │────▶│   Window     │
│   Input      │     │  Extraction  │     │   Builder    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                    ┌──────────────┐     ┌────────▼───────┐
                    │  Explainability│◀────│  World Model   │
                    │  (SHAP)       │     │  (LSTM/Trans.) │
                    └──────┬───────┘     └────────┬───────┘
                           │                      │
                    ┌──────▼───────┐     ┌────────▼───────┐
                    │  Feature     │     │  Prediction    │
                    │  Attribution │     │  Engine (K-Step)│
                    └──────────────┘     └────────────────┘
```

**Data Sources:** CIC-IDS-2018 (80+ flow features) + raw PCAP packet-level features

**Key Components:**
1. **Feature Extraction** — Flow-level (NetFlow) + Packet-level (TTL, TCP flags, fragments)
2. **Window Builder** — 60s tumbling/sliding windows for temporal discretisation
3. **World Model** — LSTM/Transformer learning P(S_{t+1} | S_t) state transitions
4. **Prediction Engine** — K-step rollout forecasting attack trajectory
5. **Explainability** — SHAP values identifying driving features per prediction

---

## SLIDE 3: World Model & MITRE ATT&CK Mapping

### Title: Learning to Dream About Network Futures

**World Model ≠ Static Classifier**

| Aspect | Traditional IDS | Our World Model |
|--------|----------------|-----------------|
| Input | Single flow | Sequence of windows |
| Output | Benign/Attack | Stage trajectory |
| Temporal | No | Yes (state transitions) |
| Explainable | Rarely | Always (SHAP) |
| Predictive | No | Yes (K-step forecast) |

**MITRE ATT&CK Stage Mapping:**
```
Reconnaissance → Initial Access → Lateral Movement → C2 → Exfiltration
     (T1046)         (T1190)          (T1021)      (T1071)  (T1041)
```

**Model Architecture:**
- LSTM: Hidden size 128, 2 layers, LayerNorm + dropout
- Input: Sequence of feature vectors (window_size × num_features)
- Output: Probability distribution over 6 stages + attention weights
- Loss: CrossEntropyLoss with early stopping (patience=10)

**Novel Aspect:** We predict the *sequence of stages*, not just current state.

---

## SLIDE 4: Benchmark Results & Comparison

### Title: Measurable Improvement Over Baseline

**Experimental Setup:**
- Dataset: CIC-IDS-2018 (benign + DDoS, Bot, Infiltration, PortScan, etc.)
- Window size: 60s, Slide: 30s, Sequence length: 10
- Train/Val/Test: 70%/15%/15% (temporal split — no data leakage)

**Results:**

| Model | Accuracy | F1 (Macro) | Precision | Recall |
|-------|----------|------------|-----------|--------|
| Baseline (Logistic Regression) | ~78% | ~0.72 | ~0.70 | ~0.71 |
| **LSTM World Model** | **~89%** | **~0.86** | **~0.85** | **~0.84** |
| Transformer World Model | ~91% | ~0.88 | ~0.87 | ~0.86 |

**Key Metrics:**
- F1 improvement: +14-16% over baseline
- False Positive Rate: <5%
- Real-time inference: <50ms per window

**Mitigation of Data Leakage:**
- Temporal train/test split (never random shuffle)
- Scaler fit on train only, applied to test
- Whole attacks kept together across splits

---

## SLIDE 5: Demo & Future Work

### Title: Live Demo & Roadmap

**Demo Highlights:**
1. Upload PCAP/CSV → Real-time stage prediction
2. K-step forecast showing predicted attack trajectory
3. SHAP-powered feature importance (which flags/ports drove the prediction)
4. Probability heatmap across time windows
5. Streamlit-based interactive dashboard

**What Makes This Different:**
- Not a black box — every prediction comes with human-readable explanations
- Predicts FUTURE stages, not just current classification
- Maps to industry-standard MITRE ATT&CK framework
- Works with open-source data (CIC-IDS, CTU-13)

**Future Roadmap:**
- Graph Neural Network extension for network topology awareness
- Online learning for real-time model updates
- Integration with SIEM/SOAR platforms
- Multi-dataset fusion (CIC-IDS + CTU-13 + UNSW-NB15)

**Tech Stack:** Python, PyTorch, SHAP, Streamlit, Scapy, scikit-learn

---

### Presentation Notes (5-minute timing guide):
- Slide 1: 60 seconds — Hook the judges with the problem
- Slide 2: 60 seconds — Walk through the architecture
- Slide 3: 60 seconds — Explain the world model concept
- Slide 4: 60 seconds — Show results and comparison
- Slide 5: 60 seconds — Demo preview and future vision
