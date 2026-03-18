"""
src/features/preprocessor.py
Data preprocessing: scaling, label encoding, train/val/test splitting,
and conversion to CNN flow matrices.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer

from src.features.extractor import FLOW_FEATURES
from src.utils.logger import get_logger

logger = get_logger("preprocessor")


class DataPreprocessor:
    """
    Handles all data preprocessing for the traffic classifier pipeline.
    - Cleans and imputes missing values
    - Scales features (StandardScaler / RobustScaler)
    - Encodes labels
    - Splits into train/val/test sets
    - Converts to CNN flow matrices
    """

    def __init__(self, scaler_type: str = "robust", random_state: int = 42):
        self.scaler_type = scaler_type
        self.random_state = random_state
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy="median")
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        self.is_fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        label_col: str = "label",
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple:
        """
        Full preprocessing pipeline on raw dataframe.
        Returns: (X_train, X_val, X_test, y_train, y_val, y_test, class_names)
        """
        feature_cols = feature_cols or FLOW_FEATURES
        
        # Only keep features that exist in dataframe
        available = [f for f in feature_cols if f in df.columns]
        missing = set(feature_cols) - set(available)
        if missing:
            logger.warning(f"Missing features (will use zeros): {missing}")
            for f in missing:
                df[f] = 0.0
        
        self.feature_names = feature_cols

        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[label_col])
        
        X = df[feature_cols].values.astype(np.float64)
        y_raw = df[label_col].values

        # Impute
        X = self.imputer.fit_transform(X)
        
        # Encode labels
        y = self.label_encoder.fit_transform(y_raw)
        self.class_names = list(self.label_encoder.classes_)
        
        logger.info(f"Classes: {self.class_names}")
        logger.info(f"Dataset shape: {X.shape}, Label distribution:")
        for cls, count in zip(*np.unique(y, return_counts=True)):
            logger.info(f"  {self.class_names[cls]}: {count}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        val_rel = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_rel, random_state=self.random_state, stratify=y_train
        )

        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        self.is_fitted = True
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test, self.class_names

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted scaler and imputer."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        X = np.where(np.isinf(X), np.nan, X)
        X = self.imputer.transform(X)
        return self.scaler.transform(X)

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe using fitted preprocessor."""
        for f in self.feature_names:
            if f not in df.columns:
                df[f] = 0.0
        X = df[self.feature_names].values.astype(np.float64)
        return self.transform(X)

    def save(self, output_dir: str):
        """Persist scaler, imputer, and encoder to disk."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        joblib.dump(self.label_encoder, f"{output_dir}/label_encoder.pkl")
        joblib.dump(self.imputer, f"{output_dir}/imputer.pkl")
        joblib.dump(self.feature_names, f"{output_dir}/feature_names.pkl")
        logger.info(f"Preprocessor saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "DataPreprocessor":
        """Load a fitted preprocessor from disk."""
        pp = cls()
        pp.scaler = joblib.load(f"{model_dir}/scaler.pkl")
        pp.label_encoder = joblib.load(f"{model_dir}/label_encoder.pkl")
        pp.imputer = joblib.load(f"{model_dir}/imputer.pkl")
        pp.feature_names = joblib.load(f"{model_dir}/feature_names.pkl")
        pp.class_names = list(pp.label_encoder.classes_)
        pp.is_fitted = True
        return pp


def to_cnn_matrix(X: np.ndarray, window_size: int = 20) -> np.ndarray:
    """
    Convert flat feature vectors to 2D flow matrices for CNN input.
    
    Groups consecutive samples into windows of `window_size` rows.
    Output shape: (n_windows, window_size, n_features, 1)
    
    This represents a 'flow image' where:
      - Rows = consecutive packets/time windows
      - Columns = feature dimensions
    """
    n_samples, n_features = X.shape
    n_windows = n_samples // window_size
    
    if n_windows == 0:
        raise ValueError(f"Not enough samples ({n_samples}) for window_size={window_size}")
    
    X_windowed = X[:n_windows * window_size].reshape(n_windows, window_size, n_features)
    X_cnn = X_windowed[..., np.newaxis]  # Add channel dim
    logger.info(f"CNN matrix shape: {X_cnn.shape}")
    return X_cnn


def labels_to_windows(y: np.ndarray, window_size: int = 20) -> np.ndarray:
    """Convert labels to match windowed samples (majority vote per window)."""
    n_windows = len(y) // window_size
    y_windows = []
    for i in range(n_windows):
        window_labels = y[i * window_size: (i + 1) * window_size]
        # Majority vote
        counts = np.bincount(window_labels)
        y_windows.append(np.argmax(counts))
    return np.array(y_windows)
