"""
src/models/supervised.py
Supervised classification: SVM and Random Forest.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.utils.logger import get_logger
from src.utils.visualizer import plot_confusion_matrix, plot_feature_importance

logger = get_logger("supervised")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    y_prob: Optional[np.ndarray] = None,
                    class_names: List[str] = None) -> Dict:
    """Compute comprehensive classification metrics."""
    avg = "weighted"
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average=avg, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"
            )
        except Exception:
            pass
    
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    metrics["classification_report"] = report
    return metrics


class SVMClassifier:
    """SVM-based traffic classifier with RBF kernel (best for non-linear flow data)."""

    def __init__(self, kernel: str = "rbf", C: float = 10.0, 
                 gamma: str = "scale", probability: bool = True,
                 max_iter: int = 2000):
        self.model = SVC(
            kernel=kernel, C=C, gamma=gamma,
            probability=probability, max_iter=max_iter,
            class_weight="balanced", random_state=42
        )
        self.class_names: List[str] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            class_names: List[str] = None) -> "SVMClassifier":
        self.class_names = class_names or [str(i) for i in np.unique(y_train)]
        logger.info(f"Training SVM on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        logger.info("SVM training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 save_dir: str = None) -> Dict:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob, self.class_names)
        
        logger.info(f"SVM Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"SVM F1 (weighted): {metrics['f1_weighted']:.4f}")
        logger.info("\n" + metrics["classification_report"])
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plot_confusion_matrix(
                y_test, y_pred, self.class_names,
                save_path=f"{save_dir}/svm_confusion_matrix.png",
                title="SVM Confusion Matrix"
            )
        return metrics

    def tune_hyperparams(self, X: np.ndarray, y: np.ndarray,
                         cv: int = 3) -> Dict:
        """Grid search for best SVM hyperparameters."""
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "kernel": ["rbf", "linear"],
        }
        logger.info("Running SVM GridSearchCV (this may take a while)...")
        gs = GridSearchCV(
            SVC(probability=True, class_weight="balanced"),
            param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1
        )
        gs.fit(X, y)
        logger.info(f"Best params: {gs.best_params_}, Score: {gs.best_score_:.4f}")
        self.model = gs.best_estimator_
        return gs.best_params_

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "class_names": self.class_names}, path)
        logger.info(f"SVM saved: {path}")

    @classmethod
    def load(cls, path: str) -> "SVMClassifier":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data["model"]
        obj.class_names = data["class_names"]
        return obj


class RandomForestTrafficClassifier:
    """
    Random Forest classifier — provides feature importance insights
    and handles class imbalance well.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 20,
                 min_samples_split: int = 5, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self.class_names: List[str] = []
        self.feature_names: List[str] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            class_names: List[str] = None,
            feature_names: List[str] = None) -> "RandomForestTrafficClassifier":
        self.class_names = class_names or [str(i) for i in np.unique(y_train)]
        self.feature_names = feature_names or [f"feat_{i}" for i in range(X_train.shape[1])]
        logger.info(f"Training Random Forest on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        
        # Log top features
        importances = self.model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:5]
        logger.info("Top 5 features:")
        for i in top_idx:
            logger.info(f"  {self.feature_names[i]}: {importances[i]:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 save_dir: str = None) -> Dict:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob, self.class_names)
        
        logger.info(f"Random Forest Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Random Forest F1 (weighted): {metrics['f1_weighted']:.4f}")
        logger.info("\n" + metrics["classification_report"])
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plot_confusion_matrix(
                y_test, y_pred, self.class_names,
                save_path=f"{save_dir}/rf_confusion_matrix.png",
                title="Random Forest Confusion Matrix"
            )
            if self.feature_names:
                plot_feature_importance(
                    self.feature_names,
                    self.model.feature_importances_,
                    save_path=f"{save_dir}/rf_feature_importance.png",
                )
        return metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        result = {"cv_mean": float(scores.mean()), "cv_std": float(scores.std())}
        logger.info(f"CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
        return result

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "class_names": self.class_names,
            "feature_names": self.feature_names,
        }, path)
        logger.info(f"Random Forest saved: {path}")

    @classmethod
    def load(cls, path: str) -> "RandomForestTrafficClassifier":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data["model"]
        obj.class_names = data["class_names"]
        obj.feature_names = data.get("feature_names", [])
        return obj


def run_supervised_pipeline(
    X_train, X_val, X_test, y_train, y_val, y_test,
    class_names: List[str],
    feature_names: List[str],
    config: dict,
    output_dir: str,
    plot_dir: str,
) -> Dict:
    """Full supervised training pipeline: SVM + Random Forest."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {}

    # ── SVM ──────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING: SVM")
    logger.info("=" * 60)
    svm_cfg = config.get("svm", {})
    svm = SVMClassifier(
        kernel=svm_cfg.get("kernel", "rbf"),
        C=svm_cfg.get("C", 10.0),
        gamma=svm_cfg.get("gamma", "scale"),
        probability=svm_cfg.get("probability", True),
        max_iter=svm_cfg.get("max_iter", 2000),
    )
    svm.fit(X_train, y_train, class_names=class_names)
    svm_metrics = svm.evaluate(X_test, y_test, save_dir=plot_dir)
    svm.save(f"{output_dir}/svm_model.pkl")
    results["svm"] = {k: v for k, v in svm_metrics.items() if k != "classification_report"}

    # ── Random Forest ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING: Random Forest")
    logger.info("=" * 60)
    rf_cfg = config.get("random_forest", {})
    rf = RandomForestTrafficClassifier(
        n_estimators=rf_cfg.get("n_estimators", 200),
        max_depth=rf_cfg.get("max_depth", 20),
        min_samples_split=rf_cfg.get("min_samples_split", 5),
        random_state=rf_cfg.get("random_state", 42),
    )
    rf.fit(X_train, y_train, class_names=class_names, feature_names=feature_names)
    rf_metrics = rf.evaluate(X_test, y_test, save_dir=plot_dir)
    rf.save(f"{output_dir}/rf_model.pkl")
    results["random_forest"] = {k: v for k, v in rf_metrics.items() if k != "classification_report"}

    # Summary
    logger.info("=" * 60)
    logger.info("SUPERVISED PIPELINE SUMMARY")
    logger.info("=" * 60)
    for model_name, m in results.items():
        logger.info(f"{model_name}: acc={m['accuracy']:.4f}, f1={m['f1_weighted']:.4f}")

    return results
