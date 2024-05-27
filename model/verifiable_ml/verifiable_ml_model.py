import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from crypten.dp import GaussianMechanism
from crypten.nn import DPLinear
from crypten.optim import DPOptimizer
from crypten.torch_utils import to_crypten_tensor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from preprocessing.data_preprocessing import preprocess_data

def load_preprocessed_data(file_path: str) -> pd.DataFrame:
    """
    Loads preprocessed data from a CSV file and returns a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The preprocessed data as a DataFrame.
    """
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
    # Define features and target
    X = data.drop(columns=["DEFAULT"]).values
    y = data["DEFAULT"].values

    return X, y

def train_logistic_regression_model(X: np.ndarray, y: np.ndarray):
    """
    Trains a logistic regression model using the given features and target.

    Args:
        X (np.ndarray): The features for model training.
        y (np.ndarray): The target variable for model training.

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluates the model using test data.

    Args:
        model: The trained logistic regression model.
        X_test (np.ndarray): The test features.
        y_test (np.ndarray): The test target variable.

    Returns:
        float: The accuracy of the model.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

class DPLogisticRegression(nn.Module):
    """
    A differentially private logistic regression model using the crypten library.
    """

    def __init__(self, input_dim: int, epsilon: float, delta: float):
        super().__init__()
        self.fc = DPLinear(input_dim, 1, epsilon=epsilon, delta=delta)

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x

def train_dp_logistic_regression_model(X: np.ndarray, y: np.ndarray, epsilon: float, delta: float):
    """
    Trains a differentially private logistic regression model using the given features and target.

    Args:
        X (np.ndarray): The features for model training.
        y (np.ndarray): The target variable for model training.
        epsilon (float): The privacy budget for differential privacy.
        delta (float): The probability of failure for differential privacy.

    Returns:
        DPLogisticRegression: The trained differentially private logistic regression model.
    """
    # Convert data to Crypten tensors
    X_ct = to_crypten_tensor(X, torch.float32)
    y_ct = to_crypten_tensor(y, torch.float32)

    # Define the model
    model = DPLogisticRegression(X.shape[1], epsilon, delta)

    # Define the loss function and optimizer
    loss_function = nn.BCELoss()
    optimizer = DPOptimizer(optim.SGD(model.parameters(), lr=0.01), epsilon, delta)

    # Train the model
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_ct)
        loss = loss_function(outputs, y_ct)
        loss.backward()
        optimizer.step()

    return model

def evaluate_dp_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluates the differentially private model using test data.

    Args:
        model: The trained differentially private logistic regression model.
        X_test (np.ndarray): The test features.
        y_test (np.ndarray): The test target variable.

    Returns:
        float: The accuracy of the model.
    """
    y_pred = (model(to_crypten_tensor(X_test, torch.float32)).get_plain_text().detach().numpy() > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

if __name__ == "__main__":
    # Load and preprocess data
    data = load_preprocessed_data("data/bank-full.csv")
    X, y = prepare_features_and_target(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model without differential privacy
    logistic_regression_model = train_logistic_regression_model(X_train, y_train)
    accuracy = evaluate_model(logistic_regression_model, X_test, y_test)
    print(f"Accuracy without differential privacy: {accuracy}")

    # Train logistic regression model with differential privacy
    epsilon = 1.0
    delta = 1e-5
    dp_logistic_regression_model = train_dp_logistic_regression_model(X_train, y_train, epsilon, delta)
    accuracy = evaluate_dp_model(dp_logistic_regression_model, X_test, y_test)
    print(f"Accuracy with differential privacy: {accuracy}")