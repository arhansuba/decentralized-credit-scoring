from ape import Contract
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from giza.agents import AgentResult, GizaAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditScoringAgent(GizaAgent):
    """
    A Giza agent that performs credit scoring using a machine learning model.

    Attributes:
        id (str): The unique ID of the agent.
        name (str): The name of the agent.
        model (sklearn.base.BaseEstimator): The machine learning model used for credit scoring.
    """

    def __init__(self, id: str, name: str, contracts, dataset_path: str = None, model=None, use_grid_search: bool = False):
        """
        Initializes a CreditScoringAgent object.

        Args:
            id (str): The unique ID of the agent.
            name (str): The name of the agent.
            contracts: Contracts for the Giza agent.
            dataset_path (str): The path to the dataset CSV file.
            model (sklearn.base.BaseEstimator): The machine learning model used for credit scoring.
            use_grid_search (bool): Flag indicating whether to use GridSearchCV for hyperparameter tuning.
        """
        super().__init__(id, name, contracts, dataset_path, use_grid_search )
        self.dataset_path = dataset_path
        self.model = model if model else LogisticRegression()
        self.use_grid_search = use_grid_search

    def preprocess_data(self, dataset: pd.DataFrame):
        """Preprocess the dataset by encoding categorical variables and scaling numeric features."""
        logger.info("Preprocessing data...")
        # Encode categorical variables
        for column in dataset.select_dtypes(include=['object']).columns:
            dataset[column] = LabelEncoder().fit_transform(dataset[column])

        # Separate features and target
        X = dataset.drop(columns=["target"])
        y = dataset["target"]

        # Scale numeric features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y

    def train_model(self):
        """Trains the machine learning model if a dataset path is provided."""
        if self.dataset_path:
            # Load the dataset
            dataset = pd.read_csv(self.dataset_path)
            X, y = self.preprocess_data(dataset)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if self.use_grid_search:
                logger.info("Using GridSearchCV for hyperparameter tuning...")
                # Define hyperparameter grid
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'liblinear']
                }
                grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                logger.info(f"Best parameters found: {grid_search.best_params_}")
            else:
                # Train the model
                logger.info("Training model...")
                self.model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)

            logger.info(f"Model evaluation metrics:\n"
                        f"Accuracy: {accuracy}\n"
                        f"Precision: {precision}\n"
                        f"Recall: {recall}\n"
                        f"F1 Score: {f1}\n"
                        f"ROC AUC Score: {roc_auc}")

    def process(self, data: pd.DataFrame) -> AgentResult:
        """
        Processes the input data by performing credit scoring using the machine learning model.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            AgentResult: The result of the agent's processing.
        """
        if self.model is None:
            raise ValueError("No model provided or trained. Please provide a model or train one with a dataset path.")

        # Preprocess the data
        X = data.copy()
        if "target" in X.columns:
            X = X.drop(columns=["target"])
        for column in X.select_dtypes(include=['object']).columns:
            X[column] = LabelEncoder().fit_transform(X[column])
        X = StandardScaler().fit_transform(X)

        # Make predictions using the machinelearning model
        predictions = self.model.predict(X)

        # Calculate the accuracy of the predictions if the actual scores are available
        accuracy = None
        if "credit_score" in data.columns:
            accuracy = accuracy_score(data["credit_score"], predictions)

        # Format the output as a string
        output = ""
        for i, prediction in enumerate(predictions):
            output += f"{data.iloc[i]['name']}: {prediction}\n"

        # Return the result as an AgentResult object
        return AgentResult(accuracy, output)


if __name__ == "__main__":
    # Contracts placeholder - replace with actual contracts as needed
    contracts = []

    # Create an instance of the CreditScoringAgent class
    agent = CreditScoringAgent(id="1", name="CreditScoringAgent", contracts=contracts, dataset_path="data/training.csv", use_grid_search=True)
    
    # Train the model if a dataset path is provided
    agent.train_model()

    # Process the data and get the result
    test_data = pd.read_csv("data/testing.csv")
    result = agent.process(test_data)

    # Print the result
    print(f"Accuracy: {result.accuracy}")
    print(f"Output:\n{result.output}")
