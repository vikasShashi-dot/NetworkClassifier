"""
src/models/cnn_model.py
CNN-based traffic classifier operating on 2D flow feature matrices.
Each 'image' represents a window of packets with features as channels.
"""
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.utils.logger import get_logger
from src.utils.visualizer import plot_confusion_matrix, plot_training_history

logger = get_logger("cnn_model")


def build_cnn_model(input_shape: Tuple, n_classes: int, 
                    filters: List[int] = None,
                    kernel_size: int = 3,
                    pool_size: int = 2,
                    dropout: float = 0.3,
                    dense_units: List[int] = None,
                    learning_rate: float = 0.001) -> Model:
    """
    Build a CNN model for flow feature matrix classification.
    
    Architecture:
    Input (window_size × n_features × 1)
      → Conv2D blocks (feature extraction)
      → GlobalAveragePooling
      → Dense layers
      → Softmax output
    
    Args:
        input_shape: (window_size, n_features, 1)
        n_classes: number of traffic classes
        filters: list of filter counts per conv block
        kernel_size: convolution kernel size
        pool_size: max-pool size
        dropout: dropout rate
        dense_units: list of dense layer sizes
        learning_rate: Adam learning rate
    """
    filters = filters or [32, 64, 128]
    dense_units = dense_units or [256, 128]

    inputs = keras.Input(shape=input_shape, name="flow_matrix")
    x = inputs

    # Convolutional blocks
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, kernel_size, padding="same", activation="relu",
                          name=f"conv_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Conv2D(f, kernel_size, padding="same", activation="relu",
                          name=f"conv_{i+1}b")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}b")(x)
        x = layers.MaxPooling2D(pool_size=(min(pool_size, x.shape[1]), 
                                          min(pool_size, x.shape[2])),
                                name=f"pool_{i+1}")(x)
        x = layers.Dropout(dropout / 2, name=f"drop_conv_{i+1}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    
    # Dense classification head
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_dense_{i+1}")(x)
        x = layers.Dropout(dropout, name=f"drop_dense_{i+1}")(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="TrafficCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class CNNTrafficClassifier:
    """
    CNN-based traffic classifier wrapper with training, evaluation, and inference.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model: Optional[Model] = None
        self.class_names: List[str] = []
        self.history = None
        self.input_shape: Optional[Tuple] = None
        self.n_classes: int = 0

    def build(self, input_shape: Tuple, n_classes: int) -> "CNNTrafficClassifier":
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model = build_cnn_model(
            input_shape=input_shape,
            n_classes=n_classes,
            filters=self.config.get("filters", [32, 64, 128]),
            kernel_size=self.config.get("kernel_size", 3),
            pool_size=self.config.get("pool_size", 2),
            dropout=self.config.get("dropout", 0.3),
            dense_units=self.config.get("dense_units", [256, 128]),
            learning_rate=self.config.get("learning_rate", 0.001),
        )
        logger.info(f"CNN model built: {input_shape} → {n_classes} classes")
        self.model.summary(print_fn=logger.info)
        return self

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        class_names: List[str] = None,
        checkpoint_dir: str = "outputs/models/cnn_checkpoints",
    ) -> "CNNTrafficClassifier":
        self.class_names = class_names or [str(i) for i in range(self.n_classes)]
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        batch_size = self.config.get("batch_size", 64)
        epochs = self.config.get("epochs", 50)
        patience = self.config.get("patience", 10)

        callbacks = [
            EarlyStopping(monitor="val_accuracy", patience=patience, 
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, 
                              min_lr=1e-6, verbose=1),
            ModelCheckpoint(
                f"{checkpoint_dir}/best_cnn.h5",
                monitor="val_accuracy", save_best_only=True, verbose=0
            ),
        ]

        logger.info(f"Training CNN: {epochs} epochs, batch_size={batch_size}")
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

        # Class weights for imbalanced data
        unique, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        class_weights = {int(cls): float(total / (len(unique) * cnt)) 
                         for cls, cnt in zip(unique, counts)}

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 save_dir: str = None) -> Dict:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_test, y_pred, 
                                        target_names=self.class_names, zero_division=0)
        
        logger.info(f"CNN Accuracy: {acc:.4f}")
        logger.info(f"CNN F1 (weighted): {f1:.4f}")
        logger.info("\n" + report)

        metrics = {"accuracy": acc, "f1_weighted": f1, "classification_report": report}

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plot_confusion_matrix(
                y_test, y_pred, self.class_names,
                save_path=f"{save_dir}/cnn_confusion_matrix.png",
                title="CNN Confusion Matrix"
            )
            if self.history:
                plot_training_history(
                    self.history,
                    save_path=f"{save_dir}/cnn_training_history.png"
                )
        return metrics

    def save(self, model_dir: str):
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        self.model.save(f"{model_dir}/cnn_model.h5")
        meta = {
            "class_names": self.class_names,
            "input_shape": list(self.input_shape),
            "n_classes": self.n_classes,
            "config": self.config,
        }
        with open(f"{model_dir}/cnn_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"CNN model saved to {model_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "CNNTrafficClassifier":
        with open(f"{model_dir}/cnn_meta.json") as f:
            meta = json.load(f)
        obj = cls(config=meta["config"])
        obj.model = keras.models.load_model(f"{model_dir}/cnn_model.h5")
        obj.class_names = meta["class_names"]
        obj.input_shape = tuple(meta["input_shape"])
        obj.n_classes = meta["n_classes"]
        logger.info(f"CNN model loaded from {model_dir}")
        return obj

    def predict_single_flow(self, flow_features: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single flow feature vector.
        flow_features: (n_features,) shaped array
        Returns: (class_name, confidence)
        """
        if self.input_shape is None:
            raise RuntimeError("Model not built.")
        window_size, n_features = self.input_shape[0], self.input_shape[1]
        
        # Pad/tile to window_size
        if len(flow_features.shape) == 1:
            flow_features = np.tile(flow_features, (window_size, 1))
        
        X = flow_features[:window_size][np.newaxis, :, :, np.newaxis]
        probs = self.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        label = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)
        return label, confidence


def run_cnn_pipeline(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    class_names: List[str],
    config: dict,
    output_dir: str,
    plot_dir: str,
    window_size: int = 20,
) -> Dict:
    """Full CNN training pipeline with windowed flow matrices."""
    from src.features.preprocessor import to_cnn_matrix, labels_to_windows
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("TRAINING: CNN on Flow Matrices")
    logger.info("=" * 60)
    
    # Convert to CNN matrices
    logger.info("Converting to flow matrices...")
    X_train_cnn = to_cnn_matrix(X_train, window_size)
    X_val_cnn = to_cnn_matrix(X_val, window_size)
    X_test_cnn = to_cnn_matrix(X_test, window_size)
    
    y_train_cnn = labels_to_windows(y_train, window_size)
    y_val_cnn = labels_to_windows(y_val, window_size)
    y_test_cnn = labels_to_windows(y_test, window_size)
    
    input_shape = X_train_cnn.shape[1:]  # (window_size, n_features, 1)
    n_classes = len(class_names)
    
    # Build and train
    cnn = CNNTrafficClassifier(config=config)
    cnn.build(input_shape, n_classes)
    cnn.fit(X_train_cnn, y_train_cnn, X_val_cnn, y_val_cnn,
            class_names=class_names,
            checkpoint_dir=f"{output_dir}/cnn_checkpoints")
    
    metrics = cnn.evaluate(X_test_cnn, y_test_cnn, save_dir=plot_dir)
    cnn.save(output_dir)
    
    return {k: v for k, v in metrics.items() if k != "classification_report"}
