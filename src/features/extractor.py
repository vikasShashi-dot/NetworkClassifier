"""
src/features/extractor.py
Extract flow-level features from PCAP files or pre-labeled CSVs.
Supports: ISCX VPN-nonVPN, UNSW-NB15, QUIC dataset
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

try:
    from scapy.all import rdpcap, IP, TCP, UDP, Raw
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger("extractor")


# ─── Flow Feature Names ───────────────────────────────────────────────────────
FLOW_FEATURES = [
    "fwd_packet_count", "bwd_packet_count",
    "total_fwd_bytes", "total_bwd_bytes",
    "fwd_pkt_len_mean", "fwd_pkt_len_std", "fwd_pkt_len_min", "fwd_pkt_len_max",
    "bwd_pkt_len_mean", "bwd_pkt_len_std", "bwd_pkt_len_min", "bwd_pkt_len_max",
    "flow_duration",
    "flow_bytes_per_sec", "flow_pkts_per_sec",
    "iat_mean", "iat_std", "iat_min", "iat_max",
    "fwd_iat_mean", "fwd_iat_std", "fwd_iat_min", "fwd_iat_max",
    "bwd_iat_mean", "bwd_iat_std", "bwd_iat_min", "bwd_iat_max",
    "fwd_bwd_byte_ratio", "fwd_bwd_pkt_ratio",
    "header_ratio", "payload_ratio",
    "active_mean", "idle_mean",
]


class FlowKey:
    """Bidirectional flow identifier (5-tuple)."""
    def __init__(self, src_ip, dst_ip, src_port, dst_port, protocol):
        # Normalize direction by sorting
        if (src_ip, src_port) > (dst_ip, dst_port):
            src_ip, dst_ip = dst_ip, src_ip
            src_port, dst_port = dst_port, src_port
        self.key = (src_ip, dst_ip, src_port, dst_port, protocol)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key


def _safe_stat(arr, fn):
    """Apply stat function safely on list/array."""
    if not arr:
        return 0.0
    return float(fn(arr))


def compute_flow_features(packets_fwd: List, packets_bwd: List, timestamps: List, protocol: int) -> Dict:
    """
    Compute flow-level features from lists of forward/backward packet sizes.
    
    Args:
        packets_fwd: list of (timestamp, size, header_size) for forward direction
        packets_bwd: list of (timestamp, size, header_size) for backward direction
        timestamps: all timestamps (sorted)
        protocol: IP protocol number
    
    Returns:
        dict of feature_name -> value
    """
    all_sizes = [p[1] for p in packets_fwd + packets_bwd]
    fwd_sizes = [p[1] for p in packets_fwd]
    bwd_sizes = [p[1] for p in packets_bwd]
    fwd_headers = [p[2] for p in packets_fwd]
    bwd_headers = [p[2] for p in packets_bwd]

    total_bytes = sum(all_sizes)
    total_fwd_bytes = sum(fwd_sizes)
    total_bwd_bytes = sum(bwd_sizes)
    total_headers = sum(fwd_headers) + sum(bwd_headers)

    ts_sorted = sorted(timestamps)
    flow_duration = (ts_sorted[-1] - ts_sorted[0]) if len(ts_sorted) > 1 else 1e-6
    flow_duration = max(flow_duration, 1e-6)

    # Inter-arrival times (all packets)
    iats = np.diff(ts_sorted).tolist() if len(ts_sorted) > 1 else [0.0]
    
    # Directional IATs
    fwd_ts = sorted([p[0] for p in packets_fwd])
    bwd_ts = sorted([p[0] for p in packets_bwd])
    fwd_iats = np.diff(fwd_ts).tolist() if len(fwd_ts) > 1 else [0.0]
    bwd_iats = np.diff(bwd_ts).tolist() if len(bwd_ts) > 1 else [0.0]

    # Active/idle time (simple heuristic: splits at IAT threshold)
    iat_threshold = 1.0  # seconds
    active_periods = [iat for iat in iats if iat < iat_threshold]
    idle_periods = [iat for iat in iats if iat >= iat_threshold]

    features = {
        "fwd_packet_count": len(packets_fwd),
        "bwd_packet_count": len(packets_bwd),
        "total_fwd_bytes": total_fwd_bytes,
        "total_bwd_bytes": total_bwd_bytes,
        # Forward packet lengths
        "fwd_pkt_len_mean": _safe_stat(fwd_sizes, np.mean),
        "fwd_pkt_len_std": _safe_stat(fwd_sizes, np.std),
        "fwd_pkt_len_min": _safe_stat(fwd_sizes, np.min),
        "fwd_pkt_len_max": _safe_stat(fwd_sizes, np.max),
        # Backward packet lengths
        "bwd_pkt_len_mean": _safe_stat(bwd_sizes, np.mean),
        "bwd_pkt_len_std": _safe_stat(bwd_sizes, np.std),
        "bwd_pkt_len_min": _safe_stat(bwd_sizes, np.min),
        "bwd_pkt_len_max": _safe_stat(bwd_sizes, np.max),
        # Flow-level
        "flow_duration": flow_duration,
        "flow_bytes_per_sec": total_bytes / flow_duration,
        "flow_pkts_per_sec": len(all_sizes) / flow_duration,
        # Inter-arrival times
        "iat_mean": _safe_stat(iats, np.mean),
        "iat_std": _safe_stat(iats, np.std),
        "iat_min": _safe_stat(iats, np.min),
        "iat_max": _safe_stat(iats, np.max),
        "fwd_iat_mean": _safe_stat(fwd_iats, np.mean),
        "fwd_iat_std": _safe_stat(fwd_iats, np.std),
        "fwd_iat_min": _safe_stat(fwd_iats, np.min),
        "fwd_iat_max": _safe_stat(fwd_iats, np.max),
        "bwd_iat_mean": _safe_stat(bwd_iats, np.mean),
        "bwd_iat_std": _safe_stat(bwd_iats, np.std),
        "bwd_iat_min": _safe_stat(bwd_iats, np.min),
        "bwd_iat_max": _safe_stat(bwd_iats, np.max),
        # Ratios
        "fwd_bwd_byte_ratio": total_fwd_bytes / max(total_bwd_bytes, 1),
        "fwd_bwd_pkt_ratio": len(packets_fwd) / max(len(packets_bwd), 1),
        "header_ratio": total_headers / max(total_bytes, 1),
        "payload_ratio": (total_bytes - total_headers) / max(total_bytes, 1),
        # Active/idle
        "active_mean": _safe_stat(active_periods, np.mean),
        "idle_mean": _safe_stat(idle_periods, np.mean),
    }
    return features


# ─── PCAP Extractor ───────────────────────────────────────────────────────────

class PcapFeatureExtractor:
    """Extract flow features from one or more PCAP files."""

    def __init__(self, flow_timeout: float = 120.0, min_packets: int = 5):
        self.flow_timeout = flow_timeout
        self.min_packets = min_packets

    def extract_from_pcap(self, pcap_path: str, label: str = "unknown") -> pd.DataFrame:
        """Process a PCAP file and return a DataFrame of flow features."""
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required for PCAP extraction. pip install scapy")

        logger.info(f"Reading PCAP: {pcap_path}")
        packets = rdpcap(pcap_path)
        return self._process_packets(packets, label)

    def _process_packets(self, packets, label: str) -> pd.DataFrame:
        flows: Dict = defaultdict(lambda: {"fwd": [], "bwd": [], "ts": [], "src": None})
        
        for pkt in tqdm(packets, desc="Processing packets", leave=False):
            if not pkt.haslayer(IP):
                continue
            ip = pkt[IP]
            proto = ip.proto
            ts = float(pkt.time)
            total_len = len(pkt)

            src_ip, dst_ip = ip.src, ip.dst
            src_port, dst_port = 0, 0
            header_size = 20  # IP header

            if ip.haslayer(TCP):
                tcp = pkt[TCP]
                src_port, dst_port = tcp.sport, tcp.dport
                header_size += tcp.dataofs * 4
            elif ip.haslayer(UDP):
                udp = pkt[UDP]
                src_port, dst_port = udp.sport, udp.dport
                header_size += 8

            fkey = FlowKey(src_ip, dst_ip, src_port, dst_port, proto)
            flow = flows[fkey]

            # Determine direction based on first seen source
            if flow["src"] is None:
                flow["src"] = src_ip

            entry = (ts, total_len, header_size)
            if src_ip == flow["src"]:
                flow["fwd"].append(entry)
            else:
                flow["bwd"].append(entry)
            flow["ts"].append(ts)

        rows = []
        for fkey, flow in flows.items():
            total_pkts = len(flow["fwd"]) + len(flow["bwd"])
            if total_pkts < self.min_packets:
                continue
            feats = compute_flow_features(flow["fwd"], flow["bwd"], flow["ts"], fkey.key[4])
            feats["label"] = label
            rows.append(feats)

        logger.info(f"Extracted {len(rows)} flows from {pcap_path}")
        return pd.DataFrame(rows)

    def extract_from_directory(self, pcap_dir: str, label_map: Dict[str, str] = None) -> pd.DataFrame:
        """
        Extract features from all PCAPs in a directory.
        label_map: {filename_substring: label} e.g. {"youtube": "streaming", "voip": "voip"}
        """
        pcap_dir = Path(pcap_dir)
        all_dfs = []
        pcap_files = list(pcap_dir.glob("**/*.pcap")) + list(pcap_dir.glob("**/*.pcapng"))
        
        if not pcap_files:
            logger.warning(f"No PCAP files found in {pcap_dir}")
            return pd.DataFrame()

        for pcap_file in tqdm(pcap_files, desc="Processing PCAPs"):
            label = "unknown"
            if label_map:
                for substr, lbl in label_map.items():
                    if substr.lower() in pcap_file.name.lower():
                        label = lbl
                        break
            try:
                df = self.extract_from_pcap(str(pcap_file), label=label)
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed on {pcap_file}: {e}")

        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)


# ─── CSV Dataset Loaders ──────────────────────────────────────────────────────

def load_unsw_nb15(csv_path: str) -> pd.DataFrame:
    """Load and map UNSW-NB15 dataset to project's feature schema."""
    logger.info(f"Loading UNSW-NB15 from {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Column mapping from UNSW-NB15 → our feature names
    column_map = {
        "spkts": "fwd_packet_count",
        "dpkts": "bwd_packet_count",
        "sbytes": "total_fwd_bytes",
        "dbytes": "total_bwd_bytes",
        "smean": "fwd_pkt_len_mean",
        "dmean": "bwd_pkt_len_mean",
        "dur": "flow_duration",
        "rate": "flow_bytes_per_sec",
        "sinpkt": "fwd_iat_mean",
        "dinpkt": "bwd_iat_mean",
        "attack_cat": "label",
    }
    df = df.rename(columns=column_map)
    
    # Fill label: NaN attack_cat = "Normal"
    if "label" in df.columns:
        df["label"] = df["label"].fillna("Normal").str.strip()
    
    # Compute derived features
    df["fwd_bwd_byte_ratio"] = df.get("total_fwd_bytes", 0) / df.get("total_bwd_bytes", 1).replace(0, 1)
    df["fwd_bwd_pkt_ratio"] = df.get("fwd_packet_count", 0) / df.get("bwd_packet_count", 1).replace(0, 1)
    df["flow_pkts_per_sec"] = (df.get("fwd_packet_count", 0) + df.get("bwd_packet_count", 0)) / df["flow_duration"].replace(0, 1e-6)
    
    # Fill missing features with 0
    for feat in FLOW_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    return df[FLOW_FEATURES + ["label"]].copy()


def load_iscx_vpn(csv_path: str) -> pd.DataFrame:
    """Load ISCX VPN-nonVPN CSV (pre-extracted features)."""
    logger.info(f"Loading ISCX VPN-nonVPN from {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # Try to find label column
    label_col = next((c for c in df.columns if "label" in c or "class" in c or "category" in c), None)
    if label_col and label_col != "label":
        df["label"] = df[label_col]
    
    # Map common ISCX columns
    col_map = {
        "total_fwd_packets": "fwd_packet_count",
        "total_backward_packets": "bwd_packet_count",
        "total_length_of_fwd_packets": "total_fwd_bytes",
        "total_length_of_bwd_packets": "total_bwd_bytes",
        "fwd_packet_length_mean": "fwd_pkt_len_mean",
        "fwd_packet_length_std": "fwd_pkt_len_std",
        "fwd_packet_length_min": "fwd_pkt_len_min",
        "fwd_packet_length_max": "fwd_pkt_len_max",
        "bwd_packet_length_mean": "bwd_pkt_len_mean",
        "bwd_packet_length_std": "bwd_pkt_len_std",
        "bwd_packet_length_min": "bwd_pkt_len_min",
        "bwd_packet_length_max": "bwd_pkt_len_max",
        "flow_duration": "flow_duration",
        "flow_bytes/s": "flow_bytes_per_sec",
        "flow_packets/s": "flow_pkts_per_sec",
        "flow_iat_mean": "iat_mean",
        "flow_iat_std": "iat_std",
        "flow_iat_min": "iat_min",
        "flow_iat_max": "iat_max",
        "fwd_iat_total": "fwd_iat_mean",
        "bwd_iat_total": "bwd_iat_mean",
    }
    df = df.rename(columns=col_map)

    for feat in FLOW_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    return df[FLOW_FEATURES + ["label"]].copy()


def load_dataset(path: str, dataset_type: str = "auto") -> pd.DataFrame:
    """
    Load a dataset from CSV or directory of PCAPs.
    dataset_type: auto | unsw_nb15 | iscx_vpn | pcap
    """
    path = Path(path)
    
    if dataset_type == "auto":
        if path.is_dir():
            dataset_type = "pcap"
        elif "unsw" in path.name.lower() or "nb15" in path.name.lower():
            dataset_type = "unsw_nb15"
        else:
            dataset_type = "iscx_vpn"

    if dataset_type == "unsw_nb15":
        return load_unsw_nb15(str(path))
    elif dataset_type == "iscx_vpn":
        return load_iscx_vpn(str(path))
    elif dataset_type == "pcap":
        extractor = PcapFeatureExtractor()
        return extractor.extract_from_directory(str(path))
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def generate_synthetic_dataset(n_samples: int = 5000, n_classes: int = 6, 
                                random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic flow feature data for testing/demonstration.
    Simulates different traffic types with realistic statistical distributions.
    """
    np.random.seed(random_state)
    
    class_configs = {
        "Web_HTTP":     {"fwd_pkt_len_mean": (500, 200), "iat_mean": (0.05, 0.02), "flow_duration": (30, 15)},
        "Streaming":    {"fwd_pkt_len_mean": (1400, 100), "iat_mean": (0.02, 0.005), "flow_duration": (120, 30)},
        "VoIP":         {"fwd_pkt_len_mean": (160, 20), "iat_mean": (0.02, 0.002), "flow_duration": (60, 20)},
        "VPN_HTTPS":    {"fwd_pkt_len_mean": (900, 300), "iat_mean": (0.1, 0.05), "flow_duration": (20, 10)},
        "P2P":          {"fwd_pkt_len_mean": (1200, 200), "iat_mean": (0.5, 0.3), "flow_duration": (300, 100)},
        "Interactive":  {"fwd_pkt_len_mean": (100, 50), "iat_mean": (0.3, 0.2), "flow_duration": (10, 5)},
    }

    rows = []
    per_class = n_samples // n_classes
    
    for label, cfg in list(class_configs.items())[:n_classes]:
        for _ in range(per_class):
            pkt_mean = max(64, np.random.normal(*cfg["fwd_pkt_len_mean"]))
            iat_mean = max(0.001, np.random.normal(*cfg["iat_mean"]))
            duration = max(1, np.random.normal(*cfg["flow_duration"]))
            fwd_pkts = np.random.randint(10, 200)
            bwd_pkts = np.random.randint(5, 100)
            total_fwd = fwd_pkts * pkt_mean
            total_bwd = bwd_pkts * pkt_mean * 0.6
            
            row = {
                "fwd_packet_count": fwd_pkts,
                "bwd_packet_count": bwd_pkts,
                "total_fwd_bytes": total_fwd,
                "total_bwd_bytes": total_bwd,
                "fwd_pkt_len_mean": pkt_mean,
                "fwd_pkt_len_std": pkt_mean * 0.2,
                "fwd_pkt_len_min": max(64, pkt_mean * 0.5),
                "fwd_pkt_len_max": min(1500, pkt_mean * 1.5),
                "bwd_pkt_len_mean": pkt_mean * 0.7,
                "bwd_pkt_len_std": pkt_mean * 0.15,
                "bwd_pkt_len_min": max(40, pkt_mean * 0.3),
                "bwd_pkt_len_max": min(1500, pkt_mean * 1.2),
                "flow_duration": duration,
                "flow_bytes_per_sec": (total_fwd + total_bwd) / duration,
                "flow_pkts_per_sec": (fwd_pkts + bwd_pkts) / duration,
                "iat_mean": iat_mean,
                "iat_std": iat_mean * 0.5,
                "iat_min": iat_mean * 0.1,
                "iat_max": iat_mean * 5,
                "fwd_iat_mean": iat_mean * 1.1,
                "fwd_iat_std": iat_mean * 0.4,
                "fwd_iat_min": iat_mean * 0.1,
                "fwd_iat_max": iat_mean * 4,
                "bwd_iat_mean": iat_mean * 0.9,
                "bwd_iat_std": iat_mean * 0.3,
                "bwd_iat_min": iat_mean * 0.1,
                "bwd_iat_max": iat_mean * 3,
                "fwd_bwd_byte_ratio": total_fwd / max(total_bwd, 1),
                "fwd_bwd_pkt_ratio": fwd_pkts / max(bwd_pkts, 1),
                "header_ratio": np.random.uniform(0.02, 0.1),
                "payload_ratio": np.random.uniform(0.9, 0.98),
                "active_mean": np.random.uniform(0.01, 0.1),
                "idle_mean": np.random.uniform(0.5, 5.0),
                "label": label,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    logger.info(f"Generated synthetic dataset: {len(df)} samples, {n_classes} classes")
    return df
