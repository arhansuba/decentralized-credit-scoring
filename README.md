# Decentralized Credit Scoring Agent








## Project Description
This project implements a decentralized credit scoring agent using machine learning and blockchain technology.

## Folder Structure
- `data/raw/`: Contains raw financial data from users.
- `data/processed/`: Contains preprocessed financial data from users.
- `preprocessing/scripts/`: Contains scripts for data preprocessing.
- `preprocessing/tools/`: Contains tools for data visualization.
- `model/credit_scoring_model/`: Contains the credit scoring model implementation.
- `model/verifiable_ml/`: Contains the verifiable ML model implementation.
- `agent/logic/`: Contains the agent's logic for predicting credit scores and executing on-chain transactions.
- `agent/giza_sdk/`: Contains the Giza SDK implementation.
- `tests/unit/`: Contains unit tests for the agent's logic.
- `tests/integration/`: Contains integration tests for the agent.

## Setup and Installation

### Prerequisites
1. Install Python 3.6 or higher from [Python Downloads](https://www.python.org/downloads/)
2. Install pip from [Pip Installation](https://pip.pypa.io/en/stable/installing/)
3. Install Git from [Git Downloads](https://git-scm.com/downloads)

### Installation Steps
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/decentralized-credit-scoring-agent.git
    cd decentralized-credit-scoring-agent
    ```
2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Agent
1. Preprocess the data:
    ```sh
    python preprocessing/scripts/data_preprocessing.py
    ```
2. Train the model:
    ```sh
    python model/credit_scoring_model/credit_scoring_model.py
    ```
3. Deploy the agent:
    ```sh
    python agent.py
    ```

## Testing the Agent
1. Run unit tests:
    ```sh
    python -m unittest discover -s tests/unit
    ```
2. Run integration tests:
    ```sh
    python -m unittest discover -s tests/integration
    ```

## Deployment
After thorough testing on a testnet, deploy the agent on a blockchain mainnet.
#   d e c e n t r a l i z e d - c r e d i t - s c o r i n g 
 
 
