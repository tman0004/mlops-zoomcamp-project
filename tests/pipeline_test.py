import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import load_data, preprocess_data, train_model, evaluate_model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow

### UNIT TESTS FOR PIPELINE MODULES ###
def test_load_data(tmp_path):
    # Create a temporary CSV file
    data = pd.DataFrame({
        "A": [1, 2],
        "B": [3, 4]
    })
    file_path = tmp_path / "test.csv"
    data.to_csv(file_path, index=False)
    df = load_data.fn(str(file_path))
    pd.testing.assert_frame_equal(df, data)

def test_preprocess_data():
    df = pd.DataFrame({
        "PassengerId": [1, 2],
        "Name": ["John Doe", "Jane Doe"],
        "Sex": ["male", "female"],
        "Age": [np.nan, 30],  # Add a non-null value for median calculation
        "Ticket": ["A/5 21171", "PC 17599"],
        "Cabin": [np.nan, "C85"],
        "Embarked": ["S", "C"],
        "Survived": [1, 0]
    })
    df["Cabin"] = df["Cabin"].astype("string")
    processed = preprocess_data.fn(df.copy()).reset_index(drop=True)
    assert not processed.empty, "Processed DataFrame is empty"
    expected_cols = {"Survived", "Age", "Sex_Encoded", "Deck_Encoded", "Embarked_Encoded"}
    assert expected_cols.issubset(set(processed.columns)), f"Missing columns: {expected_cols - set(processed.columns)}"
    row = processed.iloc[0]
    assert row["Sex_Encoded"] == 0
    assert not pd.isna(row["Age"])
    assert row["Deck_Encoded"] == 8
    assert row["Embarked_Encoded"] in [0, 1, 2]

def test_train_model(tmp_path):
    # Set up MLflow tracking for the test
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_test.db")
    mlflow.set_experiment("test_experiment")
    df = pd.DataFrame({
        "Survived": [0, 1, 0, 1],
        "Age": [22, 38, 26, 35],
        "Sex_Encoded": [0, 1, 0, 1],
        "Deck_Encoded": [8, 1, 8, 2],
        "Embarked_Encoded": [2, 0, 2, 1]
    })
    clf, X_test, y_test = train_model.fn(df)
    assert hasattr(clf, "predict")
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert len(X_test) == len(y_test)

def test_evaluate_model():
    # Simple model and data
    X = pd.DataFrame({"a": [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])
    clf = RandomForestClassifier().fit(X, y)
    acc = evaluate_model.fn(clf, X, y)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

#### INTEGRATION TESTS FOR PIPELINE MODULES ####
def test_pipeline_integration(tmp_path):
    # 1. Create a small CSV file as input
    data = pd.DataFrame({
        "PassengerId": [1, 2],
        "Name": ["John Doe", "Jane Doe"],
        "Sex": ["male", "female"],
        "Age": [22, 30],
        "Ticket": ["A/5 21171", "PC 17599"],
        "Cabin": [None, "C85"],
        "Embarked": ["S", "C"],
        "Survived": [1, 0]
    })
    file_path = tmp_path / "integration_test.csv"
    data.to_csv(file_path, index=False)

    # 2. Run the pipeline steps
    df_loaded = load_data.fn(str(file_path))
    df_processed = preprocess_data.fn(df_loaded)
    clf, X_test, y_test = train_model.fn(df_processed)
    acc = evaluate_model.fn(clf, X_test, y_test)

    # 3. Assert the pipeline runs and produces reasonable output
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    assert hasattr(clf, "predict")
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)