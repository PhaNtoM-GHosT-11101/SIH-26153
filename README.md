# SIH-26153
AI based Network Attack Forecasting from Network Traffic Data
# Challenge Specification: Predictive Cyber Defense using AI World Models

This challenge seeks software prototypes that leverage "World Models" to learn network behavior, anticipate attacker progression, and support proactive cyber defense. 

## The Core Objective
Move beyond static intrusion classification (binary benign/malicious labels) and build an AI that learns the internal causal simulation of network states over time. An infiltration is a process unfolding over time, not a single anomalous packet.

**High-Level Goals:**
* **Represent** network state using feature vectors or graphs.
* **Learn** state-transition dynamics via sequence models (LSTM, Transformer), Graph Neural Networks, or latent state models.
* **Forecast** future network states and the probability of attacker progression.
* **Map** predicted behavior to recognized frameworks (e.g., MITRE ATT&CK).
* **Explain** predictions using attention mechanisms or feature attribution.

---

## 1. Input Data Requirements

Teams must fuse two levels of traffic features drawn from open-source datasets (e.g., CIC-IDS-2018 or CTU-13) to capture both aggregate and micro-level behaviors.

| Data Level | Format Source | Extracted Features | Purpose |
| :--- | :--- | :--- | :--- |
| **Flow-Level** | NetFlow / IPFIX | IP/port pairs, TCP flags (SYN, ACK, etc.), protocol, bytes/packets per flow, duration, IAT statistics. | Captures aggregate behavior (e.g., a SYN flood or data exfiltration). |
| **Packet-Level** | PCAP-derived | TTL variance, TCP window size, IP fragment flags, payload sizes, port scan signatures, retransmissions. | Exposes timing and sequencing designed to evade basic flow-based thresholds. |

---

## 2. World Model Architecture

The core deliverable is a learned model of network state transition dynamics, not a static input-output classifier. 

* **State Representation:** Encode active flows at time $t$ into a structured feature vector or graph.
* **Transition Dynamics:** Learn the probability distribution over the next state given the current state: $P(S_{t+1} | S_t)$.
* **Model Selection:** Utilize time-windowed sequence models such as LSTMs, Temporal Transformers, or Graph Neural Networks (GNNs).
* **Training:** Use supervised dynamics learning on labeled open-source datasets, deriving ground-truth transitions from attack timelines. The model must generalize to unseen attack patterns.

---

## 3. Forward Simulation & Outputs

Given a current traffic snapshot, the prediction engine must roll out $K$ steps ahead and output:

* **Time-Series Probability Score:** The likelihood of infiltration occurring in the next $K$ time windows.
* **Predicted Attack Stage:** A direct mapping of the trajectory to MITRE ATT&CK phases (Reconnaissance, Initial Access, Lateral Movement, Command & Control, or Exfiltration).
* **Driving Features:** Interpretability tools (e.g., SHAP values, attention weights) identifying exactly which flags, ports, or temporal patterns are driving the prediction.

---

## 4. Expected Prototype Components

A software-based, fully open-source, and offline solution is required. 

* **Feature Extraction Pipeline:** Parses CSV flow records or raw PCAP files (via Scapy or PyShark) into a timestamped, normalized feature matrix.
* **Trained World Model:** Includes training scripts, model weights, and reproducible training configurations.
* **Prediction Engine:** Performs the $K$-step forward simulation from the traffic snapshot.
* **Explainability Engine:** Translates model reasoning into human-readable driving features. Black-box models are unacceptable.
* **Demonstration Interface:** An offline Streamlit, Flask, or CLI application that accepts a PCAP/CSV file, runs inference, and displays the timeline, flagged flows, and stage annotations.
* **Benchmark Results:** Proven measurable improvement (F1 score, precision, recall, false positive rate) against a baseline logistic regression model trained on identical features.

---

## 5. Deliverables for Evaluation

* **Source Code:** GitHub or Drive link.
* **Readme:** Complete local setup instructions.
* **Architecture Document:** Maximum 2 pages detailing the data flow and model structure.
* **Demo Video:** Maximum 2 minutes showing the interface in action.
* **Technical Presentation:** Maximum 5 slides summarizing the approach and benchmark results.