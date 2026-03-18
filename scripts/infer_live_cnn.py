#!/usr/bin/env python3
"""
Live traffic capture and CNN classification.
Trained on CIC-IDS-2017
Classes: Brute Force, DoS, Normal, Reconnaissance, Web Attack
Run with: sudo python scripts/infer_live_cnn.py --interface en0
"""

import argparse
import time
import threading
import queue
import numpy as np
import joblib
import json
from collections import defaultdict
from pathlib import Path

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from scapy.all import sniff, IP, TCP, UDP

MODELS_DIR   = Path("outputs/models")
WINDOW_SIZE  = 20
FLOW_TIMEOUT = 15
MIN_PACKETS  = 5

def load_models():
    print("Loading models...")
    scaler   = joblib.load(MODELS_DIR / "scaler.pkl")
    le       = joblib.load(MODELS_DIR / "label_encoder_filtered.pkl")
    features = joblib.load(MODELS_DIR / "feature_names.pkl")

    with open(MODELS_DIR / "cnn_meta.json") as f:
        meta = json.load(f)

    model_path = MODELS_DIR / "cnn_best.h5"
    if not model_path.exists():
        model_path = MODELS_DIR / "cnn_model.keras"
    cnn = keras.models.load_model(str(model_path))

    print(f"✅ CNN loaded — classes: {meta['class_names']}")
    print(f"   Features:    {len(features)}")
    print(f"   Input shape: {meta['input_shape']}")
    return cnn, scaler, le, features, meta

def extract_flow_features(packets, features):
    if len(packets) < MIN_PACKETS:
        return None

    times = np.array([p[0] for p in packets])
    sizes = np.array([p[1] for p in packets])
    dirs  = np.array([p[2] for p in packets])

    fwd_mask  = dirs == 1
    bwd_mask  = dirs == 0
    fwd_sizes = sizes[fwd_mask]
    bwd_sizes = sizes[bwd_mask]

    duration      = max(times[-1] - times[0], 1e-6)
    fp            = fwd_mask.sum()
    bp            = bwd_mask.sum()
    fwd_bytes     = fwd_sizes.sum() if len(fwd_sizes) > 0 else 0
    bwd_bytes     = bwd_sizes.sum() if len(bwd_sizes) > 0 else 0
    fwd_mean      = fwd_sizes.mean() if len(fwd_sizes) > 0 else 0
    bwd_mean      = bwd_sizes.mean() if len(bwd_sizes) > 0 else 0
    iats          = np.diff(times) if len(times) > 1 else np.array([0.0])
    fwd_iats      = np.diff(times[fwd_mask]) if fwd_mask.sum() > 1 else np.array([0.0])
    bwd_iats      = np.diff(times[bwd_mask]) if bwd_mask.sum() > 1 else np.array([0.0])
    iat_mean      = iats.mean()
    sjit          = iats.std()  if len(iats)     > 1 else 0
    djit          = bwd_iats.std() if len(bwd_iats) > 1 else 0
    total_bytes   = fwd_bytes + bwd_bytes
    total_packets = fp + bp
    header_ratio  = min((total_packets * 40) / max(total_bytes, 1), 1.0)

    row = {
        "fwd_packet_count":   fp,
        "bwd_packet_count":   bp,
        "total_fwd_bytes":    fwd_bytes,
        "total_bwd_bytes":    bwd_bytes,
        "fwd_pkt_len_mean":   fwd_mean,
        "fwd_pkt_len_std":    fwd_mean * 0.2,
        "fwd_pkt_len_min":    max(fwd_mean * 0.5, 40),
        "fwd_pkt_len_max":    min(fwd_mean * 1.5, 1500),
        "bwd_pkt_len_mean":   bwd_mean,
        "bwd_pkt_len_std":    bwd_mean * 0.2,
        "bwd_pkt_len_min":    max(bwd_mean * 0.5, 40),
        "bwd_pkt_len_max":    min(bwd_mean * 1.5, 1500),
        "flow_duration":      duration,
        "flow_bytes_per_sec": total_bytes / duration,
        "flow_pkts_per_sec":  total_packets / duration,
        "iat_mean":           iat_mean,
        "iat_std":            sjit,
        "iat_min":            iat_mean,
        "iat_max":            iat_mean + sjit * 2,
        "fwd_iat_mean":       fwd_iats.mean() if len(fwd_iats) > 0 else 0,
        "fwd_iat_std":        sjit,
        "fwd_iat_min":        fwd_iats.mean() if len(fwd_iats) > 0 else 0,
        "fwd_iat_max":        fwd_iats.mean() + sjit * 2 if len(fwd_iats) > 0 else 0,
        "bwd_iat_mean":       bwd_iats.mean() if len(bwd_iats) > 0 else 0,
        "bwd_iat_std":        djit,
        "bwd_iat_min":        bwd_iats.mean() if len(bwd_iats) > 0 else 0,
        "bwd_iat_max":        bwd_iats.mean() + djit * 2 if len(bwd_iats) > 0 else 0,
        "fwd_bwd_byte_ratio": fwd_bytes / max(bwd_bytes, 1),
        "fwd_bwd_pkt_ratio":  fp / max(bp, 1),
        "header_ratio":       header_ratio,
        "payload_ratio":      1 - header_ratio,
        "active_mean":        duration * 0.7,
        "idle_mean":          duration * 0.3,
    }

    vec = np.array([row.get(f, 0.0) for f in features], dtype=np.float64)
    return vec

def preprocess_vector(vec, scaler):
    HARD_CAPS = {
        0:  10_000,     # fwd_packet_count
        1:  10_000,     # bwd_packet_count
        13: 125_000,    # flow_bytes_per_sec
        14: 10_000,     # flow_pkts_per_sec
        2:  100_000_000,
        3:  100_000_000,
    }
    for idx, cap in HARD_CAPS.items():
        if idx < len(vec):
            vec[idx] = min(vec[idx], cap)

    # flow_duration is index 12 — convert to seconds and cap
    if 12 < len(vec):
        vec[12] = min(vec[12], 100.0)

    LOG_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                   13, 14, 15, 16, 19, 20, 23, 24, 27, 28,
                   30, 31]
    for idx in LOG_INDICES:
        if idx < len(vec):
            vec[idx] = np.log1p(max(vec[idx], 0))

    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    vec = scaler.transform(vec.reshape(1, -1))
    return vec.flatten()

class FlowBuffer:
    def __init__(self):
        self.flows = defaultdict(list)
        self.lock  = threading.Lock()

    def add_packet(self, key, ts, size, direction):
        with self.lock:
            self.flows[key].append((ts, size, direction))

    def get_expired(self):
        now     = time.time()
        expired = []
        with self.lock:
            for key, pkts in list(self.flows.items()):
                if now - pkts[-1][0] > FLOW_TIMEOUT and len(pkts) >= MIN_PACKETS:
                    expired.append((key, pkts))
                    del self.flows[key]
        return expired

    def get_all(self):
        with self.lock:
            result = [(k, v) for k, v in self.flows.items() if len(v) >= MIN_PACKETS]
            self.flows.clear()
        return result

class LiveCNNClassifier:
    def __init__(self, interface, cnn, scaler, le, features, meta):
        self.interface   = interface
        self.cnn         = cnn
        self.scaler      = scaler
        self.le          = le
        self.features    = features
        self.meta        = meta
        self.buffer      = FlowBuffer()
        self.result_q    = queue.Queue()
        self.running     = False
        self.stats       = defaultdict(int)
        self.window_size = meta.get("window_size", WINDOW_SIZE)

    def _get_flow_key(self, pkt):
        if IP not in pkt:
            return None
        src   = pkt[IP].src
        dst   = pkt[IP].dst
        proto = pkt[IP].proto
        sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
        dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
        if (src, sport) < (dst, dport):
            return (src, dst, sport, dport, proto)
        return (dst, src, dport, sport, proto)

    def _packet_callback(self, pkt):
        if IP not in pkt:
            return
        key = self._get_flow_key(pkt)
        if key is None:
            return
        direction = 1 if pkt[IP].src == key[0] else 0
        self.buffer.add_packet(key, time.time(), len(pkt), direction)
        self.stats["packets"] += 1

    def _classify_flow(self, key, packets):
        vec = extract_flow_features(packets, self.features)
        if vec is None:
            return
        vec       = preprocess_vector(vec, self.scaler)
        window    = np.tile(vec, (self.window_size, 1))
        window    = window[np.newaxis, :, :, np.newaxis]
        probs     = self.cnn.predict(window, verbose=0)[0]
        class_idx = np.argmax(probs)
        label     = self.le.classes_[class_idx]
        conf      = probs[class_idx]

        # Confidence threshold — low confidence defaults to Normal
        CLASS_THRESHOLDS = {
            "Normal":        0.85,   # high — don't want false normals
            "DoS":           0.75,   # low — missing a DoS is bad
            "Brute Force":   0.75,   # low — missing brute force is bad
            "Web Attack":    0.75,   # low — missing web attack is bad
            "Reconnaissance":0.85,   # medium — nmap scans are distinctive
        }

        threshold = CLASS_THRESHOLDS.get(label, 0.85)
        if conf < threshold:
            label = "Normal"
            conf  = 1.0 - conf

        self.result_q.put((key, label, conf, len(packets)))
        self.stats["flows"] += 1
        self.stats[label]   += 1

    def _expiry_worker(self):
        while self.running:
            time.sleep(5)
            for key, packets in self.buffer.get_expired():
                self._classify_flow(key, packets)

    def _display_worker(self):
        while self.running or not self.result_q.empty():
            try:
                key, label, conf, n_pkts = self.result_q.get(timeout=1)
                src, dst, sport, dport, proto = key
                proto_name = "TCP" if proto == 6 else ("UDP" if proto == 17 else str(proto))
                color = {
                    "Normal":        "\033[92m",   # green
                    "Reconnaissance":"\033[93m",   # yellow
                    "DoS":           "\033[91m",   # red
                    "Brute Force":   "\033[91m",   # red
                    "Web Attack":    "\033[95m",   # magenta
                }.get(label, "\033[0m")
                reset = "\033[0m"
                print(
                    f"{color}[{label:<16}]{reset} "
                    f"conf={conf:.2f} pkts={n_pkts:>4} | "
                    f"{src}:{sport} → {dst}:{dport} ({proto_name})"
                )
            except queue.Empty:
                continue

    def start(self, duration=None):
        self.running = True
        threading.Thread(target=self._expiry_worker,  daemon=True).start()
        threading.Thread(target=self._display_worker, daemon=True).start()

        print(f"\n{'='*60}")
        print(f"  Live CNN Classification — interface: {self.interface}")
        print(f"  Classes: {list(self.le.classes_)}")
        print(f"  Flow timeout: {FLOW_TIMEOUT}s  |  Min packets: {MIN_PACKETS}")
        print(f"  Duration: {'unlimited' if duration is None else f'{duration}s'}")
        print(f"  Press Ctrl+C to stop")
        print(f"{'='*60}\n")

        try:
            sniff(
                iface=self.interface,
                prn=self._packet_callback,
                filter="ip or (ip6 and tcp)",
                timeout=duration,
                store=False
            )
        except KeyboardInterrupt:
            print("\nStopping capture...")

        self.running = False
        print("Classifying remaining flows...")
        for key, packets in self.buffer.get_all():
            self._classify_flow(key, packets)

        time.sleep(2)

        print(f"\n{'='*60}")
        print(f"  CAPTURE SUMMARY")
        print(f"{'='*60}")
        print(f"  Total packets: {self.stats['packets']}")
        print(f"  Total flows:   {self.stats['flows']}")
        print(f"\n  Classification breakdown:")
        for cls in self.le.classes_:
            count = self.stats.get(cls, 0)
            bar   = "█" * min(count, 40)
            print(f"    {cls:<20} {count:>5}  {bar}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live CNN traffic classifier")
    parser.add_argument("--interface", "-i", default="en0")
    parser.add_argument("--duration",  "-d", type=int, default=None)
    args = parser.parse_args()

    cnn, scaler, le, features, meta = load_models()
    classifier = LiveCNNClassifier(
        interface=args.interface,
        cnn=cnn,
        scaler=scaler,
        le=le,
        features=features,
        meta=meta
    )
    classifier.start(duration=args.duration)
    
    

    # CNN
        #sudo python scripts/infer_live_cnn.py --interface en0