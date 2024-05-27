import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file and returns a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocesses the data by cleaning, transforming, and encoding it.

    Args:
        data (pd.DataFrame): The raw data to be preprocessed.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Preprocessed features and target as separate DataFrames.
    """
    # Drop missing values
    data = data.dropna()

    # Convert categorical variables to one-hot encoding
    data = pd.get_dummies(data, columns=['SEX', 'MARRIAGE', 'EDUCATION', 'OCCUPATION', 'HOUSING', 'LOAN'])

    # Scale numerical variables
    numerical_cols = ['AGE', 'INCOME', 'DEBT', 'ASSETS']
    data[numerical_cols] = data[numerical_cols].apply(lambda x: (x - x.mean()) / x.std())

    # Split data into features and target
    X = data.drop('CREDIT_SCORE', axis=1)
    y = data['CREDIT_SCORE']

    return X, y

def save_data(X: pd.DataFrame, y: pd.Series, X_file_path: str, y_file_path: str):
    """
    Saves preprocessed data to CSV files.

    Args:
        X (pd.DataFrame): The preprocessed features.
        y (pd.Series): The preprocessed target.
        X_file_path (str): The path to save the preprocessed features.
        y_file_path (str): The path to save the preprocessed target.
    """
    X.to_csv(X_file_path, index=False, float_format='%.4f')
    y.to_csv(y_file_path, index=False, float_format='%.4f')

def main():
    """Loads and preprocesses the data."""

    # Load data
    file_path = 'data/raw/training.csv'
    data = load_data(file_path)

    # Preprocess data
    X, y = preprocess_data(data)

    # Save preprocessed data
    X_file_path = 'data/processed/X_train.csv'
    y_file_path = 'data/processed/y_train.csv'
    save_data(X, y, X_file_path, y_file_path)

if __name__ == '__main__':
    main()