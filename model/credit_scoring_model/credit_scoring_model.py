import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

def prepare_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
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

def train_logistic_regression_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains a logistic regression model using the given features and target.

    Args:
        X (pd.DataFrame): The features for model training.
        y (pd.Series): The target variable for model training.

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using test data.

    Args:
        model (LogisticRegression): The trained logistic regression model.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.

    Returns:
        tuple[float, float, float]: A tuple containing accuracy, precision, and recall.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, _, _ = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]

    return accuracy, precision, recall

def main():
    """
    Trains and evaluates the credit scoring model.
    """
    
    # Load preprocessed data
    file_path = 'data/preprocessed/credit_data.csv'
    data = load_preprocessed_data(file_path)

    # Preprocess data
    processed_data = preprocess_data(data)

    # Prepare features and target
    X, y = prepare_features_and_target(processed_data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train logistic regression model
    model = train_logistic_regression_model(X_train, y_train)

    # Evaluate model
    accuracy, precision, recall = evaluate_model(model, X_test, y_test)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

if __name__ == "__main__":
    main()