"""
src/capture/live_capture.py
Real-time network traffic capture using Scapy + live inference.
Requires root/admin privileges and trained models.

Usage:
    sudo python scripts/infer_live.py --interface eth0 --model outputs/models/svm_model.pkl
"""
import time
import threading
import queue
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path

try:
    from scapy.all import sniff, IP, TCP, UDP, wrpcap
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[WARNING] Scapy not available. Live capture disabled.")

from src.features.extractor import FlowKey, compute_flow_features, FLOW_FEATURES
from src.features.preprocessor import DataPreprocessor
from src.utils.logger import get_logger

logger = get_logger("live_capture")


class FlowBuffer:
    """
    Maintains in-memory buffer of active flows.
    Exports flows when they are complete (timeout or FIN/RST).
    """

    def __init__(self, flow_timeout: float = 30.0, min_packets: int = 5):
        self.flow_timeout = flow_timeout
        self.min_packets = min_packets
        self.flows: Dict = defaultdict(lambda: {
            "fwd": [], "bwd": [], "ts": [], "src": None, "last_seen": 0
        })
        self._lock = threading.Lock()

    def add_packet(self, pkt) -> Optional[Dict]:
        """Add a packet to the appropriate flow. Returns flow features if flow is complete."""
        if not pkt.haslayer(IP):
            return None

        ip = pkt[IP]
        ts = float(pkt.time)
        total_len = len(pkt)
        src_ip, dst_ip = ip.src, ip.dst
        src_port, dst_port = 0, 0
        header_size = 20  # IP header
        is_fin = False

        if ip.haslayer(TCP):
            tcp = pkt[TCP]
            src_port, dst_port = tcp.sport, tcp.dport
            header_size += tcp.dataofs * 4
            if tcp.flags & 0x01 or tcp.flags & 0x04:  # FIN or RST
                is_fin = True
        elif ip.haslayer(UDP):
            udp = pkt[UDP]
            src_port, dst_port = udp.sport, udp.dport
            header_size += 8

        fkey = FlowKey(src_ip, dst_ip, src_port, dst_port, ip.proto)

        with self._lock:
            flow = self.flows[fkey]
            if flow["src"] is None:
                flow["src"] = src_ip

            entry = (ts, total_len, header_size)
            if src_ip == flow["src"]:
                flow["fwd"].append(entry)
            else:
                flow["bwd"].append(entry)
            flow["ts"].append(ts)
            flow["last_seen"] = ts

            # Export if FIN/RST
            if is_fin:
                return self._export_flow(fkey)
        return None

    def _export_flow(self, fkey: FlowKey) -> Optional[Dict]:
        """Extract and remove a flow from the buffer."""
        flow = self.flows.pop(fkey, None)
        if flow is None:
            return None
        total_pkts = len(flow["fwd"]) + len(flow["bwd"])
        if total_pkts < self.min_packets:
            return None
        try:
            features = compute_flow_features(flow["fwd"], flow["bwd"], flow["ts"], fkey.key[4])
            features["_flow_key"] = str(fkey.key)
            features["_timestamp"] = datetime.utcnow().isoformat()
            return features
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None

    def get_expired_flows(self, current_time: float) -> List[Dict]:
        """Return and remove flows that have timed out."""
        expired = []
        with self._lock:
            expired_keys = [
                k for k, v in self.flows.items()
                if current_time - v["last_seen"] > self.flow_timeout
            ]
            for k in expired_keys:
                result = self._export_flow(k)
                if result:
                    expired.append(result)
        return expired


class LiveTrafficClassifier:
    """
    Captures live traffic and classifies each flow in real-time.
    Supports any sklearn or Keras model wrapped in the project's interface.
    """

    def __init__(
        self,
        model,  # SVMClassifier | RandomForestTrafficClassifier | CNNTrafficClassifier
        preprocessor: DataPreprocessor,
        interface: str = "eth0",
        flow_timeout: float = 30.0,
        min_packets: int = 5,
        on_classification: Optional[Callable] = None,
    ):
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy required for live capture. pip install scapy")

        self.model = model
        self.preprocessor = preprocessor
        self.interface = interface
        self.buffer = FlowBuffer(flow_timeout=flow_timeout, min_packets=min_packets)
        self.on_classification = on_classification or self._default_display
        self._stop_event = threading.Event()
        self._result_queue: queue.Queue = queue.Queue()
        self.stats = {"total_flows": 0, "classified": 0, "errors": 0}
        self._results: List[Dict] = []

    def _default_display(self, result: Dict):
        """Default callback: print classification result."""
        print(
            f"[{result['timestamp']}] Flow: {result['flow_key']:<50} "
            f"→ [{result['predicted_class']:<15}] "
            f"conf={result['confidence']:.2f}"
        )

    def _classify_flow(self, features: Dict) -> Optional[Dict]:
        """Run inference on a single flow's features."""
        try:
            # Build feature vector
            X = np.array([[features.get(f, 0.0) for f in FLOW_FEATURES]])
            X = self.preprocessor.transform(X)
            
            # Handle CNN vs flat models
            if hasattr(self.model, 'predict_single_flow'):
                label, confidence = self.model.predict_single_flow(X[0])
            else:
                pred = self.model.predict(X)[0]
                if hasattr(self.model, 'predict_proba'):
                    prob = self.model.predict_proba(X)[0]
                    confidence = float(np.max(prob))
                else:
                    confidence = 1.0
                
                class_names = getattr(self.model, 'class_names', [])
                label = class_names[pred] if pred < len(class_names) else str(pred)

            return {
                "timestamp": features.get("_timestamp", datetime.utcnow().isoformat()),
                "flow_key": features.get("_flow_key", "unknown"),
                "predicted_class": label,
                "confidence": confidence,
                "flow_bytes_per_sec": features.get("flow_bytes_per_sec", 0),
                "flow_pkts_per_sec": features.get("flow_pkts_per_sec", 0),
                "flow_duration": features.get("flow_duration", 0),
            }
        except Exception as e:
            logger.warning(f"Classification error: {e}")
            self.stats["errors"] += 1
            return None

    def _packet_callback(self, pkt):
        """Scapy callback for each captured packet."""
        result = self.buffer.add_packet(pkt)
        if result:
            self.stats["total_flows"] += 1
            classified = self._classify_flow(result)
            if classified:
                self.stats["classified"] += 1
                self._result_queue.put(classified)

    def _expiry_checker(self):
        """Background thread to process timed-out flows."""
        while not self._stop_event.is_set():
            time.sleep(5)
            expired = self.buffer.get_expired_flows(time.time())
            for flow in expired:
                self.stats["total_flows"] += 1
                classified = self._classify_flow(flow)
                if classified:
                    self.stats["classified"] += 1
                    self._result_queue.put(classified)

    def _result_processor(self):
        """Background thread to process and display results."""
        while not self._stop_event.is_set() or not self._result_queue.empty():
            try:
                result = self._result_queue.get(timeout=1.0)
                self._results.append(result)
                self.on_classification(result)
            except queue.Empty:
                continue

    def start(self, duration: Optional[float] = None, packet_count: int = 0):
        """
        Start live capture.
        
        Args:
            duration: seconds to capture (None = indefinite, stop with Ctrl+C)
            packet_count: number of packets to capture (0 = unlimited)
        """
        logger.info(f"Starting live capture on {self.interface}")
        logger.info(f"Duration: {duration}s, Packet count: {packet_count or 'unlimited'}")
        logger.info("Press Ctrl+C to stop.\n")

        # Start background threads
        expiry_thread = threading.Thread(target=self._expiry_checker, daemon=True)
        result_thread = threading.Thread(target=self._result_processor, daemon=True)
        expiry_thread.start()
        result_thread.start()

        try:
            sniff(
                iface=self.interface,
                prn=self._packet_callback,
                filter="ip",
                timeout=duration,
                count=packet_count,
                store=False,
            )
        except KeyboardInterrupt:
            logger.info("Capture stopped by user.")
        finally:
            self._stop_event.set()
            result_thread.join(timeout=5)
            self._print_summary()

    def _print_summary(self):
        logger.info("\n" + "=" * 60)
        logger.info("LIVE CAPTURE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total flows detected: {self.stats['total_flows']}")
        logger.info(f"Successfully classified: {self.stats['classified']}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        if self._results:
            df = pd.DataFrame(self._results)
            logger.info("\nTraffic Class Distribution:")
            dist = df["predicted_class"].value_counts()
            for cls, cnt in dist.items():
                pct = cnt / len(df) * 100
                logger.info(f"  {cls:<20}: {cnt:>5} flows ({pct:.1f}%)")

    def get_results(self) -> pd.DataFrame:
        return pd.DataFrame(self._results)

    def save_results(self, path: str):
        df = self.get_results()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Results saved to {path}")


class TsharkCapture:
    """
    Alternative to Scapy: use tshark for capture then process PCAP.
    Useful when Scapy has compatibility issues.
    """

    def __init__(self, interface: str = "eth0", output_dir: str = "data/captures"):
        self.interface = interface
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, duration: int = 60, filter_str: str = "ip") -> str:
        """Capture traffic with tshark for `duration` seconds."""
        import subprocess
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(self.output_dir / f"capture_{timestamp}.pcap")
        
        cmd = [
            "tshark", "-i", self.interface,
            "-a", f"duration:{duration}",
            "-f", filter_str,
            "-w", output_file,
        ]
        logger.info(f"Starting tshark: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info(f"Capture saved to {output_file}")
        return output_file
