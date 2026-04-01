"""
Train a Random Forest model on the Iris dataset.

BCD4205 - Gestión de Tecnología Digital (MLOps)
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("data/iris.csv")
MODEL_PATH = Path("models/iris_model.pkl")

MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
}

TRAIN_TEST_SPLIT_PARAMS = {
    "test_size": 0.2,
    "random_state": 42,
}

TARGET_COLUMN = "species"

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Functions
# =============================================================================


def load_data(path: Path) -> pd.DataFrame:
    """Load dataset from CSV file."""
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target from dataframe."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    logger.info("Training model...")
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model


def evaluate_model(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """Evaluate model and log metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return accuracy


def save_model(model: RandomForestClassifier, path: Path) -> None:
    """Save model to pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to: {path}")


def main() -> None:
    """Execute the training pipeline."""
    logger.info("Starting training pipeline")

    df = load_data(DATA_PATH)
    X, y = split_features_target(df, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(X, y, **TRAIN_TEST_SPLIT_PARAMS)
    logger.info(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, MODEL_PATH)

    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()