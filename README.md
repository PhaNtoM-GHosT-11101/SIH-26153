# SIH-26153: AI-based Network Attack Forecasting

> **Predictive Cyber Defence using AI World Models**
> Smart India Hackathon 2026 | Challenge ID: SIH-26153

## Overview

This system uses **World Models** to learn network state-transition dynamics and predict attacker progression through MITRE ATT&CK stages — moving beyond binary detection to *predictive* cyber defense.

### Key Innovation

Instead of asking "is this an attack?", we ask **"what happens next?"** — forecasting the attacker's trajectory through stages: Reconnaissance → Initial Access → Lateral Movement → C2 → Exfiltration.

For complete documentation of every file, module, and design decision, see **[`DOCUMENTATION.md`](DOCUMENTATION.md)**.

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/PhaNtoM-GHosT-11101/SIH-26153.git
cd SIH-26153

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Get the Dataset

Download **CIC-IDS-2018** from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html). Place the CSV file in `data/raw/`.

**Alternative for quick testing** — generate the synthetic CIC-IDS-compatible dataset (temporally-structured multi-stage attack campaigns):

```bash
python tools/generate_synthetic_data.py --n-flows 120000
```

### 3. Train the Model

```bash
python train.py --data data/raw/cic-ids-2018-synthetic.csv --config configs/default.yaml
```

### 4. Run Predictions

```bash
python predict.py --data data/raw/test.csv --k-steps 5
```

### 5. Launch Demo

```bash
streamlit run src/demo/app.py
```

## Architecture

```
PCAP/CSV Input
      │
      ▼
┌──────────────────┐
│ Feature Extraction│  ← Flow-level (NetFlow) + Packet-level features
│    Pipeline       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Window Builder   │  ← 60s time windows with 30s slide
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  World Model     │  ← LSTM / Transformer / GRU
│ P(S_{t+1} | S_t)│
└────────┬─────────┘
         │
    ┌────┼────┐
    │    │    │
    ▼    ▼    ▼
 Stage  K-Step  SHAP
 Pred.  Forecast Explain
```

## Project Structure

```
SIH-26153/
├── configs/
│   └── default.yaml          # Configuration
├── src/
│   ├── data/
│   │   ├── feature_extraction.py  # CSV/PCAP parsing
│   │   ├── windowing.py          # Time window builder
│   │   └── dataset.py            # PyTorch datasets
│   ├── models/
│   │   ├── world_model.py        # LSTM/Transformer/GRU
│   │   └── baseline.py           # Logistic Regression baseline
│   ├── prediction/
│   │   └── engine.py             # K-step prediction engine
│   ├── explainability/
│   │   └── shap_explainer.py     # SHAP feature attribution
│   ├── evaluation/
│   │   └── metrics.py            # F1, precision, recall
│   └── demo/
│       └── app.py                # Streamlit dashboard
├── presentation/
│   ├── slides.md                 # 5-slide presentation content
│   ├── slides.html               # Ready-to-present slide deck
│   ├── architecture.md           # Architecture document
│   └── qa_prep.md               # Q&A preparation for judges
├── tools/
│   └── generate_synthetic_data.py  # Synthetic dataset generator
├── notes.tex / notes.pdf         # Tutorial
├── DOCUMENTATION.md              # Complete project documentation
├── train.py                      # Training script
├── predict.py                    # Inference script
├── demo.py                       # Demo data generator
└── requirements.txt
```

## Deliverables Checklist

- [x] **Source Code** — Complete Python implementation
- [x] **README** — Setup instructions (this file)
- [x] **Architecture Document** — `presentation/architecture.md`
- [x] **Complete Documentation** — `DOCUMENTATION.md`
- [ ] **Demo Video** — Recording needed (2 min max)
- [x] **Technical Presentation** — `presentation/slides.md` + `slides.html`

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch (LSTM, Transformer, GRU) |
| Feature Extraction | pandas, scikit-learn |
| Explainability | SHAP |
| Demo Interface | Streamlit + Plotly |
| Packet Parsing | Scapy |
| Baseline | Logistic Regression (scikit-learn) |

## Configuration

Edit `configs/default.yaml` to adjust:
- Window size and slide (default: 60s/30s)
- Model type (lstm/transformer/gru)
- Hyperparameters (hidden size, layers, dropout)
- Training parameters (batch size, learning rate, epochs)
- Forecast horizon (K steps)

## License

MIT License
