# 🔐 AI Network Security IDS

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)
![Datasets](https://img.shields.io/badge/Datasets-UNSW--NB15%20%7C%20CICIDS2017-purple.svg)

> AI-powered network intrusion detection system (IDS) for real-time threat detection, classification, and alerting across modern network traffic.

---

## 🎯 Project Idea

A modular, production-ready IDS leveraging machine learning and deep learning models to detect anomalies and known attack signatures in real-time using streaming network telemetry.

**Highlights:**
- Real-time packet/flow analysis with feature engineering
- Supervised and unsupervised models (XGBoost, LightGBM, Autoencoder)
- Streaming pipeline with Apache Kafka + Faust (optional)
- Explainability (SHAP) and model drift monitoring
- REST API and dashboard for alerts and metrics

---

## 🧩 Hardware/Infra Notes
- Runs on standard x86 server or cloud VM (4 vCPU, 8GB RAM minimum)
- Optional GPU for deep learning models (NVIDIA CUDA)
- NIC with port mirroring or SPAN for packet capture

---

## 🧱 Software Architecture
```
┌────────────────────────────────────────────────┐
│                 Alerting & API                 │
├────────────────────────────────────────────────┤
│  Model Serving  │  XAI (SHAP)  │  Dashboard    │
├────────────────────────────────────────────────┤
│   Feature Store │  Inference Engine │ Kafka     │
├────────────────────────────────────────────────┤
│   Capture (pcap/NetFlow/sFlow)  │  ETL         │
└────────────────────────────────────────────────┘
```

- Capture: Scapy/tshark, NetFlow/IPFIX collectors
- ETL: Pandas, PySpark (optional), streaming consumers
- Models: Sklearn/XGBoost/LightGBM, Autoencoders in PyTorch
- Serving: FastAPI + Uvicorn, Redis cache

---

## 💻 Implementation

### Dependencies & Installation
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Example requirements.txt:
```
scikit-learn
xgboost
lightgbm
pandas
numpy
scapy
pyshark
fastapi
uvicorn
pydantic
shap
pytorch-lightning
torch
faust-streaming
redis
```

### Getting Started
```bash
git clone https://github.com/AnishVyapari/ai_network_security_ids.git
cd ai_network_security_ids
python scripts/ingest_pcap.py --pcap data/sample.pcap
python scripts/train.py --dataset data/UNSW-NB15.csv --model xgb
python scripts/serve.py
```

Expose API:
```bash
uvicorn app.main:app --reload --port 8000
```

---

## 📁 File Structure
```
ai_network_security_ids/
├── app/
│   ├── main.py               # FastAPI app
│   ├── routers/              # API routes
│   └── services/             # Inference, alerts
├── scripts/
│   ├── ingest_pcap.py        # Capture/parse
│   ├── feature_engineer.py   # Feature extraction
│   ├── train.py              # Training pipeline
│   └── evaluate.py           # Metrics & reports
├── models/                   # Saved models
├── config/                   # YAML configs
├── data/                     # Datasets (gitignored)
├── tests/                    # Unit tests
├── .github/workflows/ci.yml  # CI pipeline
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧪 Testing & Datasets
- Public datasets: UNSW-NB15, CICIDS2017, NSL-KDD
- Example evaluation:
```bash
python scripts/evaluate.py --dataset data/CICIDS2017.csv --model models/xgb.bin
```

---

## 📚 Documentation Tips
- Document feature schema and normalization
- Provide dataset preparation scripts and checksums
- Include deployment recipes (Docker, systemd)
- Explain alert severity levels and thresholds

---

## ✅ CI Status
![CI](https://img.shields.io/github/actions/workflow/status/AnishVyapari/ai_network_security_ids/ci.yml?branch=main)

---

## 📄 License
MIT License. See LICENSE for details.

---

## 🙌 Acknowledgments
- Open-source IDS community
- Dataset maintainers (UNSW, CIC, NSL-KDD)
