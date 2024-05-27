import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from giza.agents import AgentResult, GizaAgent

class CreditScoringAgent(GizaAgent):
    """
    A Giza agent that performs credit scoring using a logistic regression model.

    Attributes:
        id (str): The unique ID of the agent.
        name (str): The name of the agent.
        model (LogisticRegression): The logistic regression model used for credit scoring.
    """

    def __init__(self, id: str, name: str, model: LogisticRegression):
        """
        Initializes a CreditScoringAgent object.

        Args:
            id (str): The unique ID of the agent.
            name (str): The name of the agent.
            model (LogisticRegression): The logistic regression model used for credit scoring.
        """
        super().__init__(id, name)
        self.model = model

    def process(self, data: pd.DataFrame) -> AgentResult:
        """
        Processes the input data by performing credit scoring using the logistic regression model.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            AgentResult: The result of the agent's processing.
        """
        # Make predictions using the logistic regression model
        predictions = self.model.predict(data)

        # Calculate the accuracy of the predictions
        accuracy = accuracy_score(data["credit_score"], predictions)

        # Format the output as a string
        output = ""
        for i, prediction in enumerate(predictions):
            output += f"{data.iloc[i]['name']}: {prediction}\n"

        # Return the result as an AgentResult object
        return AgentResult(accuracy, output)