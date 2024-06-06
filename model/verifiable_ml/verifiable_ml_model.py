from matplotlib import _preprocess_data
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def prepare_features_and_target(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares the features and target variables for model training.

    Args:
        data (pd.DataFrame): The preprocessed data.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the features and target variables.
    """
    logger.info("Preparing features and target...")
    X = data.drop(columns=["DEFAULT"]).values
    y = data["DEFAULT"].values
    return X, y

def train_logistic_regression_model(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """
    Trains a logistic regression model using the given features and target.

    Args:
        X (np.ndarray): The features for model training.
        y (np.ndarray): The target variable for model training.

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluates the model using test data.

    Args:
        model: The trained logistic regression model.
        X_test (np.ndarray): The test features.
        y_test (np.ndarray): The test target variable.

    Returns:
        dict: Evaluation metrics including accuracy, precision, recall, and confusion matrix.
    """
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "precision": classification_rep["weighted avg"]["precision"],
        "recall": classification_rep["weighted avg"]["recall"],
        "f1_score": classification_rep["weighted avg"]["f1-score"],
        "confusion_matrix": confusion_mat
    }
    return metrics

class SimpleNN(nn.Module):
    """
    A simple neural network for binary classification.
    """
    def __init__(self, input_dim: int):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x

def train_simple_nn_model(X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01) -> SimpleNN:
    """
    Trains a simple neural network model using the given features and target.

    Args:
        X (np.ndarray): The features for model training.
        y (np.ndarray): The target variable for model training.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.

    Returns:
        SimpleNN: The trained simple neural network model.
    """
    logger.info("Training simple neural network model...")
    model = SimpleNN(X.shape[1])
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = loss_function(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    return model

def evaluate_nn_model(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Evaluates the simple neural network model using test data.

    Args:
        model: The trained simple neural network model.
        X_test (np.ndarray): The test features.
        y_test (np.ndarray): The test target variable.

    Returns:
        float: The accuracy of the model.
    """
    logger.info("Evaluating simple neural network model...")
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy().round().astype(int)
    accuracy = accuracy_score(y_test_tensor, y_pred)
    return accuracy

if __name__ == "__main__":
    # Load and preprocess data
    data = load_preprocessed_data("data/bank-full.csv")
    processed_data = _preprocess_data(data)
    X, y = prepare_features_and_target(processed_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate logistic regression model
    logistic_regression_model = train_logistic_regression_model(X_train, y_train)
    metrics = evaluate_model(logistic_regression_model, X_test, y_test)
    logger.info(f"Evaluation of Logistic Regression model: {metrics}")

    # Train and evaluate simple neural network model
    simple_nn_model = train_simple_nn_model(X_train, y_train)
    nn_accuracy = evaluate_nn_model(simple_nn_model, X_test, y_test)
    logger.info(f"Accuracy of Simple Neural Network model: {nn_accuracy}")
