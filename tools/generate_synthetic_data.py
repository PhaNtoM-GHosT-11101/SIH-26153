import argparse
import numpy as np
import pandas as pd
from pathlib import Path

FEATURE_NAMES = [
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol",
    "Flow Duration", "Total Fwd Packets", "Total Bwd Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# Per-stage traffic fingerprints (mean, std)
PROFILES = {
    "Benign": dict(fwd_pkts=(3,1.5), bwd_pkts=(2,1.0), fwd_bytes=(1800,800), bwd_bytes=(2600,1200),
                   duration=(120000,60000), iat_mean=(30000,15000), syn=(1,0.1), ack=(3,0.8),
                   fin=(0.5,0.3), rst=(0,0.05), window=(29200,5000), ttl=(64,4),
                   ports=[(80,.3),(443,.3),(53,.15),(22,.1),(25,.05),(8080,.1)]),
    "Reconnaissance": dict(fwd_pkts=(1,0), bwd_pkts=(0.1,0.1), fwd_bytes=(48,20), bwd_bytes=(8,15),
                   duration=(800,400), iat_mean=(150,80), syn=(0.8,0.2), ack=(0.2,0.2),
                   fin=(0.1,0.1), rst=(0.5,0.3), window=(1024,512), ttl=(120,20),
                   ports=[('rand',1)]),
    "Initial Access": dict(fwd_pkts=(9,4), bwd_pkts=(7,3), fwd_bytes=(4500,2000), bwd_bytes=(5400,2600),
                   duration=(500000,250000), iat_mean=(35000,17000), syn=(2,0.7), ack=(6,2.5),
                   fin=(1.2,0.5), rst=(0.4,0.3), window=(65535,1500), ttl=(64,12),
                   ports=[(80,.3),(443,.3),(21,.2),(445,.2)]),
    "Lateral Movement": dict(fwd_pkts=(16,8), bwd_pkts=(13,6), fwd_bytes=(8200,4000), bwd_bytes=(9000,4500),
                   duration=(850000,420000), iat_mean=(42000,21000), syn=(3,1.2), ack=(11,4),
                   fin=(2,1), rst=(0.8,0.4), window=(8192,4000), ttl=(128,14),
                   ports=[(445,.4),(3389,.3),(139,.3)]),
    "Command and Control": dict(fwd_pkts=(5,2), bwd_pkts=(4,2), fwd_bytes=(2200,1100), bwd_bytes=(2600,1300),
                   duration=(2000000,900000), iat_mean=(90000,45000), syn=(1,0.4), ack=(5,2),
                   fin=(0.6,0.3), rst=(0.2,0.1), window=(65535,4000), ttl=(64,15),
                   ports=[(443,.5),(53,.3),(80,.2)]),
    "Exfiltration": dict(fwd_pkts=(3,1), bwd_pkts=(28,12), fwd_bytes=(1500,700), bwd_bytes=(150000,70000),
                   duration=(1600000,800000), iat_mean=(85000,42000), syn=(1,0.3), ack=(22,9),
                   fin=(1,0.4), rst=(0.1,0.1), window=(65535,3000), ttl=(64,10),
                   ports=[(443,.5),(80,.5)]),
}

STAGE_ORDER = ["Reconnaissance", "Initial Access", "Lateral Movement",
               "Command and Control", "Exfiltration"]


def make_flow(rng, stage, ts, src, dst):
    p = PROFILES[stage]
    if stage == "Benign":
        fwd = max(1, int(rng.normal(*p["fwd_pkts"])))
        bwd = max(0, int(rng.normal(*p["bwd_pkts"])))
    elif stage == "Reconnaissance":
        fwd = 1; bwd = 0
    else:
        fwd = max(1, int(rng.normal(*p["fwd_pkts"])))
        bwd = max(0, int(rng.normal(*p["bwd_pkts"])))

    fwd_bytes = max(0, int(rng.normal(*p["fwd_bytes"]))) * fwd
    bwd_bytes = max(0, int(rng.normal(*p["bwd_bytes"]))) * max(bwd, 1)
    duration = max(100, int(rng.normal(*p["duration"])))
    iat = max(50, int(rng.normal(*p["iat_mean"])))

    if p["ports"] == [('rand', 1)]:
        dst_port = int(rng.integers(1024, 65535))
    else:
        choices, weights = zip(*p["ports"])
        dst_port = int(rng.choice(choices, p=weights))

    syn = max(0, int(rng.normal(*p["syn"])))
    ack = max(0, int(rng.normal(*p["ack"])))
    fin = max(0, int(rng.normal(*p["fin"])))
    rst = max(0, int(rng.normal(*p["rst"])))
    window = max(1, int(rng.normal(*p["window"])))
    ttl = max(1, int(rng.normal(*p["ttl"])))

    tot_pkts = fwd + bwd
    tot_bytes = fwd_bytes + bwd_bytes
    secs = max(duration / 1e6, 0.001)
    bps = tot_bytes / secs
    pps = tot_pkts / secs
    fpps = fwd / secs
    bpps = bwd / secs
    avg_ps = tot_bytes / max(tot_pkts, 1)
    fwd_ps = fwd_bytes / max(fwd, 1)
    bwd_ps = bwd_bytes / max(bwd, 1)
    std = abs(rng.normal(avg_ps * 0.3, avg_ps * 0.1))
    iat_std = abs(rng.normal(iat * 0.5, iat * 0.2))
    iat_max = iat + abs(rng.normal(iat * 2, iat))
    iat_min = max(0, iat - abs(rng.normal(iat * 2, iat)))
    act = abs(rng.normal(duration * 0.6, duration * 0.2))
    idle = abs(rng.normal(iat * 5, iat * 2))

    return {
        "Flow ID": f"{src}-{ts*1000000}-{dst_port}-{dst}-{rst}-{fwd}",
        "Src IP": src, "Src Port": int(rng.integers(1024, 65535)),
        "Dst IP": dst, "Dst Port": dst_port,
        "Protocol": 6 if stage != "Reconnaissance" else int(rng.choice([6, 17])),
        "Flow Duration": duration, "Total Fwd Packets": fwd, "Total Bwd Packets": bwd,
        "Total Length of Fwd Packets": fwd_bytes, "Total Length of Bwd Packets": bwd_bytes,
        "Fwd Packet Length Max": fwd_ps * 1.5, "Fwd Packet Length Min": fwd_ps * 0.5,
        "Fwd Packet Length Mean": fwd_ps, "Fwd Packet Length Std": abs(rng.normal(fwd_ps*0.3, 10)),
        "Bwd Packet Length Max": bwd_ps * 1.5, "Bwd Packet Length Min": bwd_ps * 0.5,
        "Bwd Packet Length Mean": bwd_ps, "Bwd Packet Length Std": abs(rng.normal(bwd_ps*0.3, 10)),
        "Flow Bytes/s": bps, "Flow Packets/s": pps,
        "Flow IAT Mean": iat, "Flow IAT Std": iat_std, "Flow IAT Max": iat_max, "Flow IAT Min": iat_min,
        "Fwd IAT Total": iat * fwd, "Fwd IAT Mean": iat, "Fwd IAT Std": iat_std,
        "Fwd IAT Max": iat_max, "Fwd IAT Min": iat_min,
        "Bwd IAT Total": iat * bwd, "Bwd IAT Mean": iat, "Bwd IAT Std": iat_std,
        "Bwd IAT Max": iat_max, "Bwd IAT Min": iat_min,
        "Fwd PSH Flags": int(rng.choice([0,1], p=[.7,.3])), "Bwd PSH Flags": int(rng.choice([0,1], p=[.7,.3])),
        "Fwd URG Flags": 0, "Bwd URG Flags": 0,
        "Fwd Header Length": fwd*20, "Bwd Header Length": bwd*20,
        "Fwd Packets/s": fpps, "Bwd Packets/s": bpps,
        "Min Packet Length": min(fwd_ps, bwd_ps)*0.5, "Max Packet Length": max(fwd_ps, bwd_ps)*1.5,
        "Packet Length Mean": avg_ps, "Packet Length Std": std, "Packet Length Variance": std**2,
        "FIN Flag Count": fin, "SYN Flag Count": syn, "RST Flag Count": rst,
        "PSH Flag Count": int(rng.choice([0,1], p=[.7,.3])), "ACK Flag Count": ack,
        "URG Flag Count": 0, "CWE Flag Count": 0, "ECE Flag Count": 0,
        "Down/Up Ratio": bwd_bytes / max(fwd_bytes, 1), "Average Packet Size": avg_ps,
        "Avg Fwd Segment Size": fwd_ps, "Avg Bwd Segment Size": bwd_ps,
        "Fwd Header Length.1": 0 if rst > 0 else fwd*20,
        "Fwd Avg Bytes/Bulk": 0, "Fwd Avg Packets/Bulk": 0, "Fwd Avg Bulk Rate": 0,
        "Bwd Avg Bytes/Bulk": 0, "Bwd Avg Packets/Bulk": 0, "Bwd Avg Bulk Rate": 0,
        "Subflow Fwd Packets": fwd, "Subflow Fwd Bytes": fwd_bytes,
        "Subflow Bwd Packets": bwd, "Subflow Bwd Bytes": bwd_bytes,
        "Init_Win_bytes_forward": window, "Init_Win_bytes_backward": window,
        "act_data_pkt_fwd": fwd, "min_seg_size_forward": ttl,
        "Active Mean": act, "Active Std": abs(rng.normal(act*0.3, act*0.1)),
        "Active Max": act*1.5, "Active Min": act*0.5,
        "Idle Mean": idle, "Idle Std": abs(rng.normal(idle*0.3, idle*0.1)),
        "Idle Max": idle*1.5, "Idle Min": idle*0.5,
        "Timestamp": ts, "Label": stage,
    }


def generate(output, total_flows=120000, n_benign=40, n_attack=8, n_campaigns=8, seed=42):
    rng = np.random.default_rng(seed)
    print(f"Generating {total_flows} flows with {n_campaigns} attack campaigns...")

    benign_ips = [f"192.168.1.{rng.integers(10, 250)}" for _ in range(n_benign)]
    attack_ips = [f"192.168.2.{rng.integers(10, 200)}" for _ in range(n_attack)]
    ext_ips = [f"10.0.{rng.integers(0,5)}.{rng.integers(1,255)}" for _ in range(15)]

    rows = []
    ts = 0
    campaign_len = int(total_flows / (n_campaigns * 5))  # flows per stage per campaign

    for c in range(n_campaigns):
        # benign background before campaign
        bg = campaign_len * 3
        for _ in range(bg):
            rows.append(make_flow(rng, "Benign", ts, rng.choice(benign_ips), rng.choice(ext_ips)))
            ts += rng.integers(10, 60)
        # multi-stage attack campaign
        for stage in STAGE_ORDER:
            for _ in range(campaign_len):
                src = rng.choice(attack_ips)
                dst = rng.choice(ext_ips) if rng.random() < 0.7 else rng.choice(benign_ips)
                rows.append(make_flow(rng, stage, ts, src, dst))
                ts += rng.integers(2, 30)
        # benign after
        for _ in range(bg):
            rows.append(make_flow(rng, "Benign", ts, rng.choice(benign_ips), rng.choice(ext_ips)))
            ts += rng.integers(10, 60)

    df = pd.DataFrame(rows)
    df = df[FEATURE_NAMES + ["Timestamp", "Label"]]
    df = df.sort_values("Timestamp").reset_index(drop=True)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nDataset saved to {out}")
    print(f"Total flows: {len(df)} | Size: {out.stat().st_size/1e6:.1f} MB")
    print(f"Time span: {df['Timestamp'].min()}s to {df['Timestamp'].max()}s")
    print("\nStage distribution:")
    for s, cnt in df["Label"].value_counts().items():
        print(f"  {s}: {cnt} ({cnt/len(df)*100:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/raw/cic-ids-2018-synthetic.csv")
    ap.add_argument("--n-flows", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.output, args.n_flows, seed=args.seed)
