"""
scripts/evaluate.py
Evaluate a saved model on a test dataset.

Usage:
    python scripts/evaluate.py --model outputs/models/svm_model.pkl --data data/processed/features.csv
    python scripts/evaluate.py --model outputs/models/rf_model.pkl --data data/processed/features.csv
    python scripts/evaluate.py --model outputs/models --model-type cnn --data data/processed/features.csv
"""
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.extractor import load_dataset, generate_synthetic_dataset, FLOW_FEATURES
from src.features.preprocessor import DataPreprocessor
from src.utils.logger import get_logger
from src.utils.visualizer import plot_confusion_matrix

logger = get_logger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained traffic classifier")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.pkl or dir for CNN)")
    parser.add_argument("--model-type", choices=["svm", "rf", "cnn", "auto"], 
                        default="auto", help="Model type")
    parser.add_argument("--preprocessor", type=str, default="outputs/models",
                        help="Directory with saved preprocessor")
    parser.add_argument("--data", type=str, help="Path to dataset CSV")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic test data")
    parser.add_argument("--output", type=str, default="outputs/reports", help="Output directory for report")
    args = parser.parse_args()

    # Load data
    if args.synthetic:
        df = generate_synthetic_dataset(n_samples=2000)
    elif args.data:
        df = load_dataset(args.data)
    else:
        logger.error("Provide --data or --synthetic")
        sys.exit(1)

    # Load preprocessor
    try:
        pp = DataPreprocessor.load(args.preprocessor)
        logger.info(f"Preprocessor loaded from {args.preprocessor}")
    except Exception as e:
        logger.error(f"Could not load preprocessor: {e}")
        sys.exit(1)

    # Prepare data
    df = df.dropna(subset=["label"])
    X = pp.transform_df(df)
    y = pp.label_encoder.transform(df["label"].values) if "label" in df.columns else None

    # Detect model type
    model_path = args.model
    if args.model_type == "auto":
        if model_path.endswith("svm_model.pkl") or "svm" in model_path.lower():
            args.model_type = "svm"
        elif model_path.endswith("rf_model.pkl") or "rf" in model_path.lower():
            args.model_type = "rf"
        elif Path(model_path).is_dir():
            args.model_type = "cnn"
        else:
            args.model_type = "svm"

    # Load and evaluate model
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.model_type == "svm":
        from src.models.supervised import SVMClassifier
        model = SVMClassifier.load(model_path)
        metrics = model.evaluate(X, y, save_dir=args.output)
        
    elif args.model_type == "rf":
        from src.models.supervised import RandomForestTrafficClassifier
        model = RandomForestTrafficClassifier.load(model_path)
        metrics = model.evaluate(X, y, save_dir=args.output)
        
    elif args.model_type == "cnn":
        from src.models.cnn_model import CNNTrafficClassifier
        from src.features.preprocessor import to_cnn_matrix, labels_to_windows
        model = CNNTrafficClassifier.load(model_path)
        window_size = model.input_shape[0]
        X_cnn = to_cnn_matrix(X, window_size)
        y_cnn = labels_to_windows(y, window_size)
        metrics = model.evaluate(X_cnn, y_cnn, save_dir=args.output)

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Accuracy:    {metrics.get('accuracy', 0):.4f}")
    logger.info(f"F1 (weighted): {metrics.get('f1_weighted', 0):.4f}")
    if "classification_report" in metrics:
        logger.info("\n" + metrics["classification_report"])


if __name__ == "__main__":
    main()
