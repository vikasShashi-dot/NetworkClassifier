"""
scripts/infer_live.py
Live traffic capture + real-time classification.

Requires:
    - Trained model (--model)
    - Root/admin privileges (for packet capture)
    - Scapy installed

Usage:
    sudo python scripts/infer_live.py --interface eth0
    sudo python scripts/infer_live.py --interface eth0 --model outputs/models/rf_model.pkl --model-type rf
    sudo python scripts/infer_live.py --interface eth0 --duration 60
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.preprocessor import DataPreprocessor
from src.capture.live_capture import LiveTrafficClassifier
from src.utils.logger import get_logger

logger = get_logger("infer_live")


def main():
    parser = argparse.ArgumentParser(description="Live Network Traffic Classifier")
    parser.add_argument("--interface", type=str, default="eth0", help="Network interface to capture on")
    parser.add_argument("--model", type=str, default="outputs/models/svm_model.pkl",
                        help="Path to trained model")
    parser.add_argument("--model-type", choices=["svm", "rf", "cnn"], default="svm",
                        help="Type of model to use")
    parser.add_argument("--preprocessor", type=str, default="outputs/models",
                        help="Directory with fitted preprocessor")
    parser.add_argument("--duration", type=float, default=None,
                        help="Capture duration in seconds (default: indefinite)")
    parser.add_argument("--min-packets", type=int, default=5,
                        help="Minimum packets per flow to classify")
    parser.add_argument("--flow-timeout", type=float, default=30.0,
                        help="Seconds of inactivity to expire a flow")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Save classification results to CSV")
    args = parser.parse_args()

    # Load preprocessor
    logger.info(f"Loading preprocessor from {args.preprocessor}...")
    try:
        pp = DataPreprocessor.load(args.preprocessor)
    except Exception as e:
        logger.error(f"Failed to load preprocessor: {e}")
        logger.info("Train a model first: python scripts/train.py --mode all --synthetic")
        sys.exit(1)

    # Load model
    logger.info(f"Loading {args.model_type} model from {args.model}...")
    try:
        if args.model_type == "svm":
            from src.models.supervised import SVMClassifier
            model = SVMClassifier.load(args.model)
        elif args.model_type == "rf":
            from src.models.supervised import RandomForestTrafficClassifier
            model = RandomForestTrafficClassifier.load(args.model)
        elif args.model_type == "cnn":
            from src.models.cnn_model import CNNTrafficClassifier
            model = CNNTrafficClassifier.load(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Start live classifier
    classifier = LiveTrafficClassifier(
        model=model,
        preprocessor=pp,
        interface=args.interface,
        flow_timeout=args.flow_timeout,
        min_packets=args.min_packets,
    )

    logger.info("\n" + "=" * 70)
    logger.info("LIVE TRAFFIC CLASSIFICATION")
    logger.info("=" * 70)
    logger.info(f"Interface:    {args.interface}")
    logger.info(f"Model:        {args.model_type.upper()} → {args.model}")
    logger.info(f"Classes:      {model.class_names}")
    logger.info("=" * 70 + "\n")

    classifier.start(duration=args.duration)

    if args.save_results:
        classifier.save_results(args.save_results)
        logger.info(f"Results saved to: {args.save_results}")


if __name__ == "__main__":
    main()
