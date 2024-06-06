from matplotlib import _preprocess_data
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

file_path = 'data/training.csv'

def load_preprocessed_data(file_path: str) -> pd.DataFrame:
    """
    Loads preprocessed data from a CSV file and returns a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The preprocessed data as a DataFrame.
    """
    logger.info(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    return data

def prepare_features_and_target(data: pd.DataFrame) -> tuple:
    """
    Prepares the features and target variables for model training.

    Args:
        data (pd.DataFrame): The preprocessed data.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the features and target variables.
    """
    # Define features and target
    X = data.drop(columns=["DEFAULT"])
    y = data["DEFAULT"]
    return X, y

def train_logistic_regression_model(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """
    Trains a logistic regression model using the given features and target.

    Args:
        X (pd.DataFrame): The features for model training.
        y (pd.Series): The target variable for model training.

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def train_random_forest_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Trains a random forest model using the given features and target.

    Args:
        X (pd.DataFrame): The features for model training.
        y (pd.Series): The target variable for model training.

    Returns:
        RandomForestClassifier: The trained random forest model.
    """
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluates the model using test data.

    Args:
        model: The trained machine learning model.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.

    Returns:
        dict: A dictionary containing various evaluation metrics.
    """
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["precision"],
        "recall": classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["recall"],
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return metrics

def main():
    """
    Trains and evaluates the credit scoring model.
    """
    # Load preprocessed data
    training_file_path = 'data/training.csv'
    testing_file_path = 'data/testing.csv'
    hasil_file_path = 'data/Hasil Testing Credit Scoring.csv'

    training_data = load_preprocessed_data(training_file_path)
    testing_data = load_preprocessed_data(testing_file_path)
    hasil_data = load_preprocessed_data(hasil_file_path)

    # Preprocess data
    training_processed_data = _preprocess_data(training_data)
    testing_processed_data = _preprocess_data(testing_data)
    hasil_processed_data = _preprocess_data(hasil_data)

    # Prepare features and target
    X_train, y_train = prepare_features_and_target(training_processed_data)
    X_test, y_test = prepare_features_and_target(testing_processed_data)
    X_hasil, y_hasil = prepare_features_and_target(hasil_processed_data)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train logistic regression model
    lr_model = train_logistic_regression_model(X_train, y_train)

    # Train random forest model
    rf_model = train_random_forest_model(X_train, y_train)

    # Evaluate models
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    # Log evaluation metrics
    logger.info("Logistic Regression Metrics:")
    for metric, value in lr_metrics.items():
        logger.info(f"{metric.capitalize()}: {value}")

    logger.info("Random Forest Metrics:")
    for metric, value in rf_metrics.items():
        logger.info(f"{metric.capitalize()}: {value}")

if __name__ == "__main__":
    main()