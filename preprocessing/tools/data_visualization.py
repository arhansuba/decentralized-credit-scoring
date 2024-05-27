import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def visualize_correlation(data: pd.DataFrame):
    """
    Visualizes the correlation between variables in the data.

    Args:
        data (pd.DataFrame): The data to calculate the correlation matrix for.
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def visualize_distribution(data: pd.DataFrame, features: list):
    """
    Visualizes the distribution of the specified features in the data.

    Args:
        data (pd.DataFrame): The data containing the features.
        features (list): A list of feature names to visualize.
    """
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.show()

def visualize_pair_plot(data: pd.DataFrame):
    """
    Visualizes pair plots of the data to understand relationships between variables.

    Args:
        data (pd.DataFrame): The data to visualize pair plots for.
    """
    plt.figure(figsize=(12, 10))
    sns.pairplot(data, diag_kind='kde')
    plt.show()

def main():
    """Visualizes the data."""
    
    # Load data
    file_path = 'data/raw/training.csv'
    data = load_data(file_path)

    # Visualize correlation
    visualize_correlation(data)

    # Visualize distribution
    features = ['AGE', 'INCOME', 'DEBT', 'ASSETS']
    visualize_distribution(data, features)

    # Visualize pair plots
    visualize_pair_plot(data)

if __name__ == '__main__':
    main()