"""
scripts/train.py
Main CLI entry point for training the traffic classifier.

Usage:
    # Use synthetic data (for testing):
    python scripts/train.py --mode all --synthetic

    # Use real dataset:
    python scripts/train.py --mode all --data data/processed/features.csv

    # Extract features from PCAP first:
    python scripts/train.py --mode extract --input data/raw/ --output data/processed/features.csv

    # Train only supervised models:
    python scripts/train.py --mode supervised --data data/processed/features.csv

    # Train only CNN:
    python scripts/train.py --mode cnn --data data/processed/features.csv
"""
import sys
import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.extractor import load_dataset, generate_synthetic_dataset, FLOW_FEATURES
from src.features.preprocessor import DataPreprocessor
from src.models.unsupervised import run_unsupervised_pipeline
from src.models.supervised import run_supervised_pipeline
from src.models.cnn_model import run_cnn_pipeline
from src.utils.logger import get_logger
from src.utils.visualizer import plot_class_distribution

logger = get_logger("train")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_features(args, config):
    """Extract features from PCAP or raw CSV."""
    from src.features.extractor import PcapFeatureExtractor
    
    logger.info(f"Extracting features from: {args.input}")
    input_path = Path(args.input)
    
    if input_path.is_dir():
        extractor = PcapFeatureExtractor(
            flow_timeout=config["features"]["flow_timeout"],
            min_packets=config["features"]["min_packets"],
        )
        df = extractor.extract_from_directory(str(input_path))
    else:
        df = load_dataset(str(input_path), dataset_type=args.dataset_type)
    
    if df.empty:
        logger.error("No data extracted. Check your input path.")
        sys.exit(1)
    
    output_path = args.output or "data/processed/features.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Features saved to: {output_path} ({len(df)} flows)")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Network Traffic Classifier - Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--mode", choices=["extract", "unsupervised", "supervised", "cnn", "all"],
                        default="all", help="Training mode")
    parser.add_argument("--data", type=str, help="Path to feature CSV file")
    parser.add_argument("--input", type=str, help="Input PCAP dir or raw CSV (for extract mode)")
    parser.add_argument("--output", type=str, help="Output path for extracted features")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--dataset-type", choices=["auto", "unsw_nb15", "iscx_vpn", "pcap"],
                        default="auto", help="Dataset type for feature loading")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (for testing without a real dataset)")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Number of synthetic samples (when --synthetic)")
    parser.add_argument("--no-cnn", action="store_true", help="Skip CNN training")

    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not Path(config_path).exists():
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {}
    else:
        config = load_config(config_path)

    # Output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = config.get("outputs", {}).get("model_dir", "outputs/models")
    plot_dir = config.get("outputs", {}).get("plot_dir", f"outputs/plots/{timestamp}")
    report_dir = config.get("outputs", {}).get("report_dir", "outputs/reports")
    
    for d in [model_dir, plot_dir, report_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Feature Extraction ────────────────────────────────────────────────────
    if args.mode == "extract":
        extract_features(args, config)
        return

    # ── Load Data ─────────────────────────────────────────────────────────────
    if args.synthetic:
        logger.info("Using SYNTHETIC dataset for demonstration")
        df = generate_synthetic_dataset(
            n_samples=args.n_samples,
            n_classes=6,
            random_state=config.get("dataset", {}).get("random_state", 42)
        )
    elif args.data:
        df = load_dataset(args.data, dataset_type=args.dataset_type)
    else:
        logger.warning("No data provided. Using synthetic data. Use --data <path> or --synthetic.")
        df = generate_synthetic_dataset(n_samples=3000)

    if df.empty:
        logger.error("DataFrame is empty. Cannot train.")
        sys.exit(1)

    logger.info(f"Loaded dataset: {df.shape[0]} samples")
    
    # Plot class distribution
    if "label" in df.columns:
        plot_class_distribution(
            df["label"].values,
            save_path=f"{plot_dir}/class_distribution.png"
        )

    # ── Preprocessing ─────────────────────────────────────────────────────────
    logger.info("Preprocessing data...")
    ds_cfg = config.get("dataset", {})
    pp = DataPreprocessor(random_state=ds_cfg.get("random_state", 42))
    
    feature_cols = config.get("features", {}).get("selected", FLOW_FEATURES)
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = pp.fit_transform(
        df,
        feature_cols=feature_cols,
        label_col=ds_cfg.get("label_column", "label"),
        test_size=ds_cfg.get("test_size", 0.2),
        val_size=ds_cfg.get("val_size", 0.1),
    )
    pp.save(model_dir)

    all_results = {}

    # ── Unsupervised ──────────────────────────────────────────────────────────
    if args.mode in ("unsupervised", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: UNSUPERVISED LEARNING")
        logger.info("=" * 60)
        unsup_results = run_unsupervised_pipeline(
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=pp.feature_names,
            class_names=class_names,
            config=config.get("unsupervised", {}),
            output_dir=model_dir,
            plot_dir=plot_dir,
        )
        all_results["unsupervised"] = unsup_results

    # ── Supervised ────────────────────────────────────────────────────────────
    if args.mode in ("supervised", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: SUPERVISED LEARNING")
        logger.info("=" * 60)
        sup_results = run_supervised_pipeline(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            class_names=class_names,
            feature_names=pp.feature_names,
            config=config.get("supervised", {}),
            output_dir=model_dir,
            plot_dir=plot_dir,
        )
        all_results["supervised"] = sup_results

    # ── CNN ───────────────────────────────────────────────────────────────────
    if args.mode in ("cnn", "all") and not args.no_cnn:
        try:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 3: CNN DEEP LEARNING")
            logger.info("=" * 60)
            cnn_results = run_cnn_pipeline(
                X_train=X_train, X_val=X_val, X_test=X_test,
                y_train=y_train, y_val=y_val, y_test=y_test,
                class_names=class_names,
                config=config.get("cnn", {}),
                output_dir=model_dir,
                plot_dir=plot_dir,
                window_size=config.get("features", {}).get("window_size", 20),
            )
            all_results["cnn"] = cnn_results
        except Exception as e:
            logger.warning(f"CNN training failed: {e} (this is OK, other models still trained)")

    # ── Save results report ───────────────────────────────────────────────────
    report_path = f"{report_dir}/training_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nTraining report saved to: {report_path}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    if "supervised" in all_results:
        for model_name, metrics in all_results["supervised"].items():
            logger.info(f"{model_name:>20}: acc={metrics.get('accuracy', 0):.4f}, "
                        f"f1={metrics.get('f1_weighted', 0):.4f}")
    if "cnn" in all_results:
        cnn_m = all_results["cnn"]
        logger.info(f"{'cnn':>20}: acc={cnn_m.get('accuracy', 0):.4f}, "
                    f"f1={cnn_m.get('f1_weighted', 0):.4f}")
    
    logger.info(f"\nModels saved to: {model_dir}")
    logger.info(f"Plots saved to:  {plot_dir}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
