import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from prefect import flow, task
from sklearn.base import BaseEstimator

@task
def load_data(file_name: str):
    # Load the dataset from a CSV file
    df = pd.read_csv(file_name)    
    return df

@task
def preprocess_data(df: pd.DataFrame):
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
    # Split data
    X = preprocessed_df.drop('Survived', axis=1)
    y = preprocessed_df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Log parameters
    for param_name, param_value in clf.get_params().items():
        mlflow.log_param(param_name, param_value)

    # Log model
    mlflow.sklearn.log_model(clf, "model")

    return clf, X_test, y_test

@task
def evaluate_model(clf: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

@flow
def ml_pipeline():
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('titanic_experiment')
    
    with mlflow.start_run(): 
        df = load_data('data/train.csv')
        preprocessed_df = preprocess_data(df)
        clf, X_test, y_test = train_model(preprocessed_df)
        accuracy = evaluate_model(clf, X_test, y_test)
        mlflow.log_metric("accuarcy", accuracy)

if __name__ == "__main__":
    ml_pipeline()