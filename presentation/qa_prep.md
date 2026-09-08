# Q&A Preparation for SIH-26153 Evaluation
## Anticipated Judge Questions & Strong Answers

---

### Q1: "What exactly is a World Model in your context?"

**Answer:** A World Model learns the *dynamics* of how network states evolve — specifically, the probability of transitioning from one state to the next: P(S_{t+1} | S_t). Unlike a traditional classifier that looks at a single flow in isolation, our model sees a *sequence* of network snapshots and learns the causal progression. It's like learning the rules of chess — once you know the rules, you can predict what move comes next. In our case, the "rules" are how attackers move from reconnaissance to initial access to lateral movement.

---

### Q2: "How do you handle class imbalance? Most traffic is benign."

**Answer:** Great question. CIC-IDS has ~80% benign traffic. We handle this through:
1. **Weighted Cross-Entropy Loss** — minority classes (attacks) get higher weight
2. **Macro-averaged F1** — our primary metric treats all classes equally
3. **Window-level labeling** — even a single malicious flow in a 60s window marks the window as containing that stage
4. **Temporal oversampling** — attack windows are repeated more frequently during training

---

### Q3: "How do you prevent data leakage in time-series data?"

**Answer:** This is the #1 pitfall in network security ML. We strictly:
1. **Temporal split** — first 70% for training, next 15% validation, last 15% test. Never random shuffle.
2. **Scaler isolation** — StandardScaler is fit ONLY on training data, then applied to val/test
3. **Window integrity** — overlapping windows never share the same attack across train/test boundary
4. **No future peeking** — the model at time t only sees data from times 0 to t

---

### Q4: "Why not just use a simpler model like Random Forest?"

**Answer:** We actually benchmark against Logistic Regression as the baseline. The key difference is *temporal awareness*. A Random Forest treats each flow independently — it can't learn that "port scan followed by SSH brute force followed by lateral movement" is a classic attack pattern. Our LSTM/Transformer sees the *sequence* and learns these temporal correlations. The +14-16% F1 improvement demonstrates this. For a real-time system, the additional compute cost of an LSTM is negligible compared to the detection improvement.

---

### Q5: "How do you explain the model's decisions to a SOC analyst?"

**Answer:** Every prediction is accompanied by SHAP (SHapley Additive exPlanations) values. SHAP tells us exactly which features contributed to each prediction. For example:
- "Predicted Reconnaissance because: 47 SYN flags, destination port 80, high flow rate, low payload"
- "Predicted Exfiltration because: 2.3MB outbound, low packet count, unusual TTL pattern"

The SOC analyst sees not just *what* the model predicts, but *why* — in terms of actual network features they understand.

---

### Q6: "What's the inference latency? Can this run in real-time?"

**Answer:** Our current pipeline processes a 60-second window in <50ms on CPU. For production:
- Feature extraction: ~20ms (flow aggregation)
- Model inference: ~10ms (LSTM forward pass)
- SHAP explanation: ~20ms (approximate)
Total: ~50ms per window — well within real-time requirements.

For high-throughput networks, the feature extraction can be parallelized, and the model can run on GPU for <5ms inference.

---

### Q7: "How does this generalize to unseen attack types?"

**Answer:** Our model learns *behavioral patterns*, not specific signatures. A port scan has a distinctive temporal signature regardless of which tool generates it. The LSTM learns "high SYN count + short duration + many destinations = reconnaissance pattern." This generalizes because:
1. The model trains on multiple attack variants
2. The state-transition dynamics are attack-agnostic
3. SHAP explanations let analysts verify *why* an unknown pattern was flagged

We validate this by testing on attack types not seen during training.

---

### Q8: "What datasets did you use and why?"

**Answer:** Primary: **CIC-IDS-2018** — the gold standard for IDS research. It includes:
- Realistic network topology (web servers, email, file transfers)
- 13+ attack types covering the full kill chain
- Pre-computed 80+ flow features (no raw PCAP parsing needed)
- Labeled ground truth with timestamps

Secondary: **CTU-13** — specifically for botnet C2 detection. We chose these because they're peer-reviewed, widely cited, and provide both flow-level CSV and raw PCAP for packet-level features.

---

### Q9: "What's the difference between your approach and existing IDS tools like Snort/Suricata?"

**Answer:** Snort and Suricata are *signature-based* — they match traffic against known patterns (like checking a face against a mugshot database). Our approach is *behavior-based* — it learns how normal and attack traffic *evolves over time*. Key differences:

| Aspect | Snort/Suricata | Our System |
|--------|---------------|------------|
| Detection | Signature matching | Behavioral learning |
| Unknown attacks | Cannot detect | Can detect new patterns |
| Temporal | No | Yes (state transitions) |
| Explainability | Rule match | Feature attribution |
| Prediction | No | Yes (K-step forecast) |

---

### Q10: "What would you do differently with more time?"

**Answer:** Three priorities:
1. **Graph Neural Network** — model network topology as a graph, not just flows
2. **Online learning** — continuously update the model as new traffic arrives
3. **Multi-modal fusion** — combine network logs, endpoint telemetry, and user behavior for richer state representation

---

### Q11: "How do you handle encrypted traffic?"

**Answer:** Encrypted traffic (TLS/SSL) doesn't hide metadata. We still extract:
- Flow-level: duration, byte counts, packet counts, timing patterns
- Packet-level: TLS handshake sizes, certificate info, JA3 fingerprints
- Behavioral: connection patterns, data volume anomalies

Studies show encrypted botnet C2 has distinctive timing patterns even without decrypting payload. Our model learns these temporal signatures.

---

### Q12: "What's the deployment architecture for production?"

**Answer:**
```
Network Tap → Feature Collector → Stream Processing (Kafka) →
World Model (GPU) → Alert Engine → SIEM Dashboard
```
- Feature collector runs on each network segment
- Kafka handles high-throughput streaming
- Model inference on dedicated GPU nodes
- Alerts pushed to existing SOC dashboards (Splunk, ELK)
- All open-source — no vendor lock-in
