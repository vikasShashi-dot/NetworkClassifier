"""tests/test_pipeline.py - Unit tests for the traffic classifier pipeline."""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_df():
    from src.features.extractor import generate_synthetic_dataset
    return generate_synthetic_dataset(n_samples=500, n_classes=4, random_state=42)

@pytest.fixture
def preprocessed_data(synthetic_df):
    from src.features.preprocessor import DataPreprocessor
    pp = DataPreprocessor(random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = pp.fit_transform(
        synthetic_df, test_size=0.2, val_size=0.1
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names, pp


# ─── Feature Extractor Tests ──────────────────────────────────────────────────

class TestFeatureExtractor:
    def test_synthetic_generation(self, synthetic_df):
        assert len(synthetic_df) > 0
        assert "label" in synthetic_df.columns
        assert synthetic_df["label"].nunique() == 4

    def test_flow_features_present(self, synthetic_df):
        from src.features.extractor import FLOW_FEATURES
        for feat in FLOW_FEATURES:
            assert feat in synthetic_df.columns, f"Missing feature: {feat}"

    def test_compute_flow_features(self):
        from src.features.extractor import compute_flow_features
        fwd = [(0.0, 1000, 40), (0.05, 800, 40), (0.1, 1200, 40)]
        bwd = [(0.02, 500, 40), (0.08, 600, 40)]
        ts = [0.0, 0.02, 0.05, 0.08, 0.1]
        feats = compute_flow_features(fwd, bwd, ts, proto=6)
        assert feats["fwd_packet_count"] == 3
        assert feats["bwd_packet_count"] == 2
        assert feats["flow_duration"] > 0
        assert feats["flow_bytes_per_sec"] > 0
        assert 0 <= feats["header_ratio"] <= 1

    def test_no_nan_in_synthetic(self, synthetic_df):
        from src.features.extractor import FLOW_FEATURES
        assert not synthetic_df[FLOW_FEATURES].isnull().all().any()


# ─── Preprocessor Tests ───────────────────────────────────────────────────────

class TestPreprocessor:
    def test_output_shapes(self, preprocessed_data):
        X_train, X_val, X_test, y_train, y_val, y_test, class_names, pp = preprocessed_data
        assert X_train.ndim == 2
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]

    def test_label_encoding(self, preprocessed_data):
        *_, y_train, y_val, y_test, class_names, pp = preprocessed_data
        assert len(class_names) == 4
        assert y_train.max() < len(class_names)
        assert y_train.min() >= 0

    def test_no_nan_after_preprocessing(self, preprocessed_data):
        X_train, X_val, X_test, *_ = preprocessed_data
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_cnn_matrix_conversion(self, preprocessed_data):
        from src.features.preprocessor import to_cnn_matrix, labels_to_windows
        X_train, _, X_test, y_train, _, y_test, _, _ = preprocessed_data
        
        window_size = 10
        n_features = X_train.shape[1]
        X_cnn = to_cnn_matrix(X_train, window_size)
        y_cnn = labels_to_windows(y_train, window_size)
        
        assert X_cnn.ndim == 4
        assert X_cnn.shape[1] == window_size
        assert X_cnn.shape[2] == n_features
        assert X_cnn.shape[3] == 1
        assert len(y_cnn) == X_cnn.shape[0]


# ─── Unsupervised Model Tests ─────────────────────────────────────────────────

class TestUnsupervised:
    def test_kmeans_fit_predict(self, preprocessed_data):
        from src.models.unsupervised import KMeansClassifier
        X_train, _, X_test, y_train, _, y_test, _, _ = preprocessed_data
        
        km = KMeansClassifier(n_clusters=4, n_init=5, max_iter=100)
        km.fit(X_train)
        labels = km.predict(X_test)
        
        assert len(labels) == len(X_test)
        assert labels.min() >= 0
        assert labels.max() < 4

    def test_kmeans_evaluate(self, preprocessed_data):
        from src.models.unsupervised import KMeansClassifier
        X_train, _, X_test, _, _, y_test, _, _ = preprocessed_data
        km = KMeansClassifier(n_clusters=4, n_init=5, max_iter=100)
        km.fit(X_train)
        metrics = km.evaluate(X_test, y_true=y_test)
        assert "silhouette_score" in metrics
        assert "adjusted_rand_index" in metrics

    def test_dbscan_fit(self, preprocessed_data):
        from src.models.unsupervised import DBSCANClassifier
        X_train, *_ = preprocessed_data
        db = DBSCANClassifier(eps=1.0, min_samples=5)
        db.fit(X_train[:200])  # Use subset
        assert db.labels_ is not None
        assert len(db.labels_) == 200


# ─── Supervised Model Tests ───────────────────────────────────────────────────

class TestSupervised:
    def test_svm_fit_predict(self, preprocessed_data):
        from src.models.supervised import SVMClassifier
        X_train, _, X_test, y_train, _, y_test, class_names, _ = preprocessed_data
        
        svm = SVMClassifier(kernel="rbf", C=1.0, max_iter=500)
        svm.fit(X_train, y_train, class_names=class_names)
        preds = svm.predict(X_test)
        
        assert len(preds) == len(X_test)
        acc = (preds == y_test).mean()
        assert acc > 0.3  # Should do better than random

    def test_rf_fit_predict(self, preprocessed_data):
        from src.models.supervised import RandomForestTrafficClassifier
        X_train, _, X_test, y_train, _, y_test, class_names, _ = preprocessed_data
        
        rf = RandomForestTrafficClassifier(n_estimators=20, max_depth=5)
        rf.fit(X_train, y_train, class_names=class_names)
        preds = rf.predict(X_test)
        
        assert len(preds) == len(X_test)
        acc = (preds == y_test).mean()
        assert acc > 0.3

    def test_rf_feature_importance(self, preprocessed_data):
        from src.models.supervised import RandomForestTrafficClassifier
        X_train, _, _, y_train, _, _, class_names, pp = preprocessed_data
        
        rf = RandomForestTrafficClassifier(n_estimators=10)
        rf.fit(X_train, y_train, class_names=class_names, feature_names=pp.feature_names)
        
        importances = rf.model.feature_importances_
        assert len(importances) == X_train.shape[1]
        assert importances.sum() > 0.99  # Should sum to ~1


# ─── CNN Model Tests ──────────────────────────────────────────────────────────

class TestCNNModel:
    def test_build_model(self):
        from src.models.cnn_model import build_cnn_model
        model = build_cnn_model(
            input_shape=(10, 20, 1),
            n_classes=4,
            filters=[16, 32],
            dense_units=[64, 32]
        )
        assert model is not None
        assert model.output_shape == (None, 4)

    def test_cnn_fit_predict(self, preprocessed_data):
        from src.models.cnn_model import CNNTrafficClassifier
        from src.features.preprocessor import to_cnn_matrix, labels_to_windows
        
        X_train, X_val, X_test, y_train, y_val, y_test, class_names, _ = preprocessed_data
        window_size = 10
        
        X_tr_cnn = to_cnn_matrix(X_train, window_size)
        X_va_cnn = to_cnn_matrix(X_val, window_size)
        X_te_cnn = to_cnn_matrix(X_test, window_size)
        y_tr_cnn = labels_to_windows(y_train, window_size)
        y_va_cnn = labels_to_windows(y_val, window_size)
        y_te_cnn = labels_to_windows(y_test, window_size)
        
        cnn = CNNTrafficClassifier(config={
            "filters": [16, 32], "kernel_size": 3, "pool_size": 2,
            "dropout": 0.2, "dense_units": [64], "learning_rate": 0.01,
            "epochs": 3, "batch_size": 32, "patience": 2
        })
        cnn.build(X_tr_cnn.shape[1:], len(class_names))
        cnn.fit(X_tr_cnn, y_tr_cnn, X_va_cnn, y_va_cnn, class_names=class_names)
        
        preds = cnn.predict(X_te_cnn)
        assert len(preds) == len(y_te_cnn)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
