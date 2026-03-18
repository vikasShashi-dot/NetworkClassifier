# Network Traffic Classification System
> Classify encrypted network traffic by application/type without reading packet content.

## Project Overview
This project implements a full ML pipeline for classifying network traffic using **flow-level features** only — no packet payload inspection. It supports both **offline dataset training** (Google Colab) and **live traffic inference** (VS Code / local).

## Architecture
```
network_traffic_classifier/
├── configs/                  # YAML configs for models and features
├── data/
│   ├── raw/                  # Place downloaded datasets here
│   └── processed/            # Feature-engineered CSVs
├── notebooks/
│   └── colab_training.ipynb  # Full Colab training notebook
├── src/
│   ├── features/             # Flow feature extraction
│   │   ├── extractor.py      # PCAP → flow features
│   │   └── preprocessor.py   # Scaling, encoding, splitting
│   ├── models/               # ML model definitions
│   │   ├── unsupervised.py   # K-Means, DBSCAN
│   │   ├── supervised.py     # SVM, Random Forest
│   │   └── cnn_model.py      # CNN on flow feature matrices
│   ├── capture/              # Live traffic capture
│   │   └── live_capture.py   # Scapy-based real-time inference
│   └── utils/                # Shared utilities
│       ├── logger.py
│       └── visualizer.py
├── scripts/
│   ├── train.py              # CLI training entry point
│   ├── evaluate.py           # Model evaluation
│   └── infer_live.py         # Live capture + inference
├── tests/                    # Unit tests
├── requirements.txt
├── requirements_colab.txt
└── setup.py
```

## Quick Start

### 1. Install Dependencies (VS Code / Local)
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
- **ISCX VPN-nonVPN**: https://www.unb.ca/cic/datasets/vpn.html
- **UNSW-NB15**: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Place CSV/PCAP files in `data/raw/`

### 3. Extract Features from PCAP
```bash
python scripts/train.py --mode extract --input data/raw/ --output data/processed/features.csv
```

### 4. Train Models
```bash
# Unsupervised first
python scripts/train.py --mode unsupervised --data data/processed/features.csv

# Supervised
python scripts/train.py --mode supervised --data data/processed/features.csv --model svm
python scripts/train.py --mode supervised --data data/processed/features.csv --model cnn

# All models
python scripts/train.py --mode all --data data/processed/features.csv
```

### 5. Evaluate
```bash
python scripts/evaluate.py --model outputs/models/svm_model.pkl --data data/processed/features.csv
```

### 6. Live Capture + Inference (requires root/admin)
```bash
sudo python scripts/infer_live.py --interface eth0 --model outputs/models/svm_model.pkl
```

## Google Colab Training
Upload `notebooks/colab_training.ipynb` to Google Colab and run all cells. The notebook handles dataset download, feature extraction, training, and saves models to Google Drive.

## Datasets Supported
| Dataset | Format | Classes |
|---------|--------|---------|
| ISCX VPN-nonVPN | PCAP + CSV | VPN vs Non-VPN, App type |
| QUIC Dataset | PCAP | QUIC protocol apps |
| UNSW-NB15 | CSV | Attack types + Normal |

## Models
| Model | Type | Use Case |
|-------|------|----------|
| K-Means | Unsupervised | Discovery of natural clusters |
| DBSCAN | Unsupervised | Anomaly/outlier detection |
| SVM | Supervised | High-accuracy classification |
| Random Forest | Supervised | Feature importance + accuracy |
| CNN | Supervised | Spatial patterns in flow matrices |

## Features Extracted
- Packet inter-arrival times (mean, std, min, max)
- Packet size statistics (mean, std, min, max, total bytes)
- Flow duration
- Forward/backward byte ratios
- Packet count (fwd/bwd)
- Bytes per second, packets per second
- Header ratio, payload ratio
