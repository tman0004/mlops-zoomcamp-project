'''
This module implements an end-to-end machine learning pipeline for the Titanic
dataset using Prefect for orchestration and MLflow for experiment tracking.
The pipeline includes data loading, preprocessing, model training with a RandomForestClassifier,
evaluation, and logging of parameters and metrics.
The code is structured with Prefect tasks and flows, and supports experiment
reproducibility and tracking via MLflow.

Main functionalities:
- Load Titanic dataset from a CSV file.
- Preprocess data: encode categorical variables, handle missing values, and drop irrelevant columns.
- Train a RandomForestClassifier and log model parameters and artifacts to MLflow.
- Evaluate the trained model and log accuracy metrics.
- Orchestrate the workflow using Prefect's task and flow abstractions.

Intended for use in MLOps workflows and experiment tracking environments.
'''
import os

import mlflow
import pandas as pd
import toml
from mlflow.entities import SourceType
from prefect import flow, task
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# secrets = toml.load(".streamlit/secrets.toml")
# aws_access_key_id = secrets["AWS_ACCESS_KEY_ID"]
# aws_secret_access_key = secrets["AWS_SECRET_ACCESS_KEY"]

# # Set environment variables for AWS credentials
# os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
# os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key


@task
def load_data(file_name: str):
    """
    Loads a dataset from a CSV file into a pandas DataFrame.

    Args:
        file_name (str): The path to the CSV file to be loaded.

    Returns:
        pandas.DataFrame: The loaded dataset as a DataFrame.
    """
    # Load the dataset from a CSV file
    df = pd.read_csv(file_name)
    return df


@task
def preprocess_data(df: pd.DataFrame):
    """
    Preprocesses the input Titanic DataFrame by encoding categorical variables, handling missing
    values, and dropping unnecessary columns.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing Titanic passenger data.
    Returns:
        pd.DataFrame: The preprocessed DataFrame with encoded features and irrelevant columns
        removed.
    Processing steps:
        - Encodes the 'Sex' column as 'Sex_Encoded' (male: 0, female: 1).
        - Fills missing values in the 'Age' column with the median age.
        - Extracts the deck letter from the 'Cabin' column, fills missing values with 'U',
          and encodes as 'Deck_Encoded'.
        - Fills missing values in the 'Embarked' column with the mode and encodes as
          'Embarked_Encoded' (C: 0, Q: 1, S: 2).
        - Drops the columns: 'PassengerId', 'Name', 'Ticket', 'Cabin', 'Deck',
          'Embarked', and 'Sex'.
    """
    # encode gender
    df['Sex_Encoded'] = df['Sex'].map({'male': 0, 'female': 1})

    # fill missing values for 'Age' with the median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # create deck feature from 'Cabin'
    df['Deck'] = df['Cabin'].str[0].fillna('U')  # 'U' for unknown
    df['Deck_Encoded'] = df['Deck'].map({
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
        'F': 5, 'G': 6, 'T': 7, 'U': 8
    })

    # fill missing values for 'Embarked' with the mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # encode 'Embarked'
    df['Embarked_Encoded'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # drop unnecessary columns
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Deck', 'Embarked', 'Sex'])

    return df


@task
def train_model(preprocessed_df: pd.DataFrame):
    """
    Trains a RandomForestClassifier on the provided preprocessed DataFrame, logs model
    parameters and the trained model to MLflow, and returns the trained classifier
    along with test data.
    Args:
        preprocessed_df (pd.DataFrame): The preprocessed DataFrame containing features
        and the target column 'Survived'.
    Returns:
        tuple: A tuple containing:
            - clf (RandomForestClassifier): The trained RandomForestClassifier model.
            - x_test (pd.DataFrame): The test set features.
            - y_test (pd.Series): The test set target values.
    """
    # Split data
    x = preprocessed_df.drop('Survived', axis=1)
    y = preprocessed_df['Survived']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_train, y_train)

    # Log parameters
    for param_name, param_value in clf.get_params().items():
        mlflow.log_param(param_name, param_value)

    # Log model
    mlflow.sklearn.log_model(clf, "model")

    return clf, x_test, y_test


@task
def evaluate_model(clf: BaseEstimator, x_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates a trained classifier on test data and returns the accuracy score.
    Parameters:
        clf (BaseEstimator): The trained classifier implementing a `predict` method.
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.
    Returns:
        float: Accuracy score of the classifier on the test data.
    """
    # Predict and evaluate
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


@flow
def ml_pipeline():
    """
    Runs the end-to-end machine learning pipeline for the Titanic
    dataset, including data loading, preprocessing,
    model training, evaluation, and experiment tracking with MLflow.
    Steps:
        1. Sets the MLflow tracking URI and experiment name.
        2. Loads the training data from a CSV file.
        3. Preprocesses the loaded data.
        4. Trains a classification model on the preprocessed data.
        5. Evaluates the trained model on the test set.
        6. Logs the evaluation metric (accuracy) to MLflow.
    Returns:
        None
    """
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('titanic_experiment2')
    with mlflow.start_run():
        df = load_data('data/train.csv')
        preprocessed_df = preprocess_data(df)
        clf, x_test, y_test = train_model(preprocessed_df)
        accuracy = evaluate_model(clf, x_test, y_test)
        mlflow.log_metric("accuarcy", accuracy)


if __name__ == "__main__":
    ml_pipeline()
