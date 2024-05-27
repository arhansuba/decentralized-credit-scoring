from giza.agents import AgentResult, GizaAgent
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class MyAgent(GizaAgent):
    """
    A custom agent that performs logistic regression on a dataset.
    """

    def __init__(self, dataset_path: str):
        """
        Initializes the agent with a dataset path.

        Args:
            dataset_path (str): The path to the dataset CSV file.
        """
        super().__init__()
        self.dataset_path = dataset_path

    def process(self, data: pd.DataFrame) -> AgentResult:
        """
        Processes the data by performing logistic regression and returns the result.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            AgentResult: The result of the logistic regression model.
        """
        # Load the dataset
        dataset = pd.read_csv(self.dataset_path)

        # Split the data into features and target
        X = dataset.drop(columns=["target"])
        y = dataset["target"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Create an AgentResult object with the accuracy
        result = AgentResult(accuracy=accuracy)

        return result

if __name__ == "__main__":
    # Create an instance of the MyAgent class
    agent = MyAgent("data/dataset.csv")

    # Process the data and get the result
    result = agent.process(pd.DataFrame())

    # Print the result
    print(f"Accuracy: {result.accuracy}")