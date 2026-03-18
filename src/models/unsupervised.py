"""
src/models/unsupervised.py
Unsupervised clustering: K-Means and DBSCAN for traffic pattern discovery.
Run these FIRST to understand natural data clusters before supervised learning.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.decomposition import PCA

from src.utils.logger import get_logger
from src.utils.visualizer import plot_clusters_2d, plot_elbow_curve

logger = get_logger("unsupervised")


class KMeansClassifier:
    """
    K-Means clustering for traffic classification.
    Supports auto-selection of k via elbow method.
    """

    def __init__(self, n_clusters: int = 8, max_iter: int = 500, 
                 n_init: int = 20, random_state: int = 42):
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "KMeansClassifier":
        logger.info(f"Fitting K-Means with k={self.n_clusters}")
        self.model.fit(X)
        self.labels_ = self.model.labels_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict:
        labels = self.predict(X)
        metrics = {}
        
        # Internal metrics (no ground truth needed)
        if len(np.unique(labels)) > 1:
            metrics["silhouette_score"] = silhouette_score(X, labels, sample_size=min(5000, len(X)))
            metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
            metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
        
        metrics["n_clusters"] = len(np.unique(labels))
        metrics["inertia"] = self.model.inertia_
        
        # External metrics (if ground truth available)
        if y_true is not None:
            metrics["adjusted_rand_index"] = adjusted_rand_score(y_true, labels)
            metrics["nmi"] = normalized_mutual_info_score(y_true, labels)
        
        logger.info("K-Means Evaluation:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
        return metrics

    def find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 16),
                       plot_path: str = None) -> int:
        """Find optimal k using elbow method and silhouette scores."""
        logger.info(f"Running elbow analysis for k in {list(k_range)}")
        inertias = []
        silhouettes = []
        
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(X, labels, sample_size=min(3000, len(X)))
                silhouettes.append(sil)
            else:
                silhouettes.append(0)

        if plot_path:
            plot_elbow_curve(list(k_range), inertias, save_path=plot_path)
        
        # Pick k with highest silhouette score
        best_k = list(k_range)[np.argmax(silhouettes)]
        logger.info(f"Optimal k by silhouette: {best_k} (score: {max(silhouettes):.4f})")
        return best_k

    def get_cluster_profiles(self, X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """Describe each cluster by feature means."""
        labels = self.predict(X)
        df = pd.DataFrame(X, columns=feature_names)
        df["cluster"] = labels
        return df.groupby("cluster").mean().round(4)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"K-Means saved: {path}")

    @classmethod
    def load(cls, path: str) -> "KMeansClassifier":
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj.n_clusters = obj.model.n_clusters
        obj.labels_ = None
        return obj


class DBSCANClassifier:
    """
    DBSCAN clustering — density-based, no need to specify k.
    Great for anomaly/outlier detection (cluster -1 = noise/anomaly).
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 10, metric: str = "euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
        self.labels_: Optional[np.ndarray] = None
        self._X_fit: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "DBSCANClassifier":
        logger.info(f"Fitting DBSCAN (eps={self.eps}, min_samples={self.min_samples})")
        self.labels_ = self.model.fit_predict(X)
        self._X_fit = X
        
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = (self.labels_ == -1).sum()
        logger.info(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points ({n_noise/len(X)*100:.1f}%)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """DBSCAN doesn't support predict; use nearest-neighbor assignment."""
        if self._X_fit is None:
            raise RuntimeError("DBSCAN must be fitted first.")
        from sklearn.neighbors import KNeighborsClassifier
        if not hasattr(self, "_knn"):
            self._knn = KNeighborsClassifier(n_neighbors=1)
            self._knn.fit(self._X_fit, self.labels_)
        return self._knn.predict(X)

    def evaluate(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict:
        labels = self.labels_ if self.labels_ is not None else self.predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        metrics = {
            "n_clusters": n_clusters,
            "n_noise_points": int(n_noise),
            "noise_ratio": float(n_noise / len(labels)),
        }
        
        valid = labels != -1
        if valid.sum() > 10 and len(np.unique(labels[valid])) > 1:
            metrics["silhouette_score"] = silhouette_score(
                X[valid], labels[valid], sample_size=min(5000, valid.sum())
            )
        
        if y_true is not None:
            metrics["adjusted_rand_index"] = adjusted_rand_score(y_true, labels)
            metrics["nmi"] = normalized_mutual_info_score(y_true, labels)
        
        logger.info("DBSCAN Evaluation:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
        return metrics

    def tune_eps(self, X: np.ndarray, eps_range: List[float] = None) -> float:
        """Find good eps using k-nearest neighbor distance plot heuristic."""
        from sklearn.neighbors import NearestNeighbors
        
        eps_range = eps_range or [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
        logger.info("Tuning DBSCAN eps...")
        
        nn = NearestNeighbors(n_neighbors=self.min_samples)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        k_distances = np.sort(distances[:, -1])
        
        # Find the elbow in k-distances
        diffs = np.diff(k_distances)
        elbow_idx = np.argmax(diffs)
        suggested_eps = float(k_distances[elbow_idx])
        logger.info(f"Suggested eps from k-distance elbow: {suggested_eps:.4f}")
        return suggested_eps

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "labels": self.labels_, "X_fit": self._X_fit}, path)
        logger.info(f"DBSCAN saved: {path}")

    @classmethod
    def load(cls, path: str) -> "DBSCANClassifier":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data["model"]
        obj.labels_ = data["labels"]
        obj._X_fit = data["X_fit"]
        obj.eps = obj.model.eps
        obj.min_samples = obj.model.min_samples
        return obj


def run_unsupervised_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    config: dict,
    output_dir: str,
    plot_dir: str,
) -> Dict:
    """
    Full unsupervised pipeline:
    1. Find optimal k for K-Means
    2. Fit K-Means and DBSCAN
    3. Evaluate and visualize
    4. Save models
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    results = {}

    # ── K-Means ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1: K-Means Clustering")
    logger.info("=" * 60)
    
    km_cfg = config.get("kmeans", {})
    km = KMeansClassifier(
        n_clusters=km_cfg.get("n_clusters", 8),
        max_iter=km_cfg.get("max_iter", 500),
        n_init=km_cfg.get("n_init", 20),
        random_state=km_cfg.get("random_state", 42),
    )
    
    # Optional: auto-find k
    # optimal_k = km.find_optimal_k(X_train, k_range=range(2, 15),
    #                               plot_path=f"{plot_dir}/kmeans_elbow.png")
    # km.n_clusters = optimal_k

    km.fit(X_train)
    km_metrics = km.evaluate(X_test, y_true=y_test)
    results["kmeans"] = km_metrics
    
    # Cluster profiles
    profiles = km.get_cluster_profiles(X_test, feature_names)
    profiles.to_csv(f"{output_dir}/kmeans_cluster_profiles.csv")
    
    # Visualization
    labels_pred = km.predict(X_test)
    plot_clusters_2d(X_test, labels_pred, method="pca",
                     save_path=f"{plot_dir}/kmeans_clusters_pca.png",
                     title="K-Means Clusters")
    plot_clusters_2d(X_test, y_test, method="pca",
                     save_path=f"{plot_dir}/true_labels_pca.png",
                     title="True Labels")
    
    km.save(f"{output_dir}/kmeans_model.pkl")

    # ── DBSCAN ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2: DBSCAN Clustering")
    logger.info("=" * 60)
    
    db_cfg = config.get("dbscan", {})
    db = DBSCANClassifier(
        eps=db_cfg.get("eps", 0.5),
        min_samples=db_cfg.get("min_samples", 10),
        metric=db_cfg.get("metric", "euclidean"),
    )
    
    # Use subset for DBSCAN (can be slow on large data)
    max_dbscan_samples = 10000
    if len(X_train) > max_dbscan_samples:
        idx = np.random.choice(len(X_train), max_dbscan_samples, replace=False)
        X_db = X_train[idx]
    else:
        X_db = X_train
    
    # Auto-tune eps if needed
    # suggested_eps = db.tune_eps(X_db)
    
    db.fit(X_db)
    db_metrics = db.evaluate(X_db, y_true=None)
    results["dbscan"] = db_metrics
    
    plot_clusters_2d(X_db, db.labels_, method="pca",
                     save_path=f"{plot_dir}/dbscan_clusters_pca.png",
                     title="DBSCAN Clusters")
    
    db.save(f"{output_dir}/dbscan_model.pkl")

    logger.info(f"Unsupervised pipeline complete. Results: {results}")
    return results
