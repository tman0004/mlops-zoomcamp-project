"""
Unit and integration tests for the pipeline module.
This test suite covers the following pipeline components:
- Data loading (`load_data`)
- Data preprocessing (`preprocess_data`)
- Model training (`train_model`)
- Model evaluation (`evaluate_model`)
Tests include:
- Unit tests for each pipeline step, verifying correct functionality and output types.
- Integration test that runs the full pipeline on a small sample dataset to 
  ensure end-to-end correctness.
Temporary files and MLflow tracking are used to isolate test runs.
"""
import sys
import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.ensemble import RandomForestClassifier
from pipeline import load_data, preprocess_data, train_model, evaluate_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
### UNIT TESTS FOR PIPELINE MODULES ###
def test_load_data(tmp_path):
    """
    Test the `load_data` function to ensure it correctly loads a CSV file into a pandas DataFrame.
    This test creates a temporary CSV file with sample data, uses the `load_data.fn` 
    function to load the file,
    and asserts that the loaded DataFrame matches the original data.
    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest for file operations.
    Raises:
        AssertionError: If the loaded DataFrame does not match the original data.
    """
    data = pd.DataFrame({
        "A": [1, 2],
        "B": [3, 4]
    })
    file_path = tmp_path / "test.csv"
    data.to_csv(file_path, index=False)
    df = load_data.fn(str(file_path))
    pd.testing.assert_frame_equal(df, data)

def test_preprocess_data():
    """
    Test the `preprocess_data.fn` function to ensure it correctly processes \
    a sample Titanic DataFrame.
    This test checks that:
    - The processed DataFrame is not empty.
    - The expected columns ('Survived', 'Age', 'Sex_Encoded', 'Deck_Encoded', 
      'Embarked_Encoded') are present.
    - The 'Sex_Encoded' value for the first row is 0 (corresponding to 'male').
    - The 'Age' value for the first row is not NaN (ensuring missing values are imputed).
    - The 'Deck_Encoded' value for the first row is 8 (corresponding to missing or unknown deck).
    - The 'Embarked_Encoded' value for the first row is within the expected range [0, 1, 2].
    """
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
    assert expected_cols.issubset(
        set(processed.columns)), f"Missing columns: {expected_cols - set(processed.columns)}"
    row = processed.iloc[0]
    assert row["Sex_Encoded"] == 0
    assert not pd.isna(row["Age"])
    assert row["Deck_Encoded"] == 8
    assert row["Embarked_Encoded"] in [0, 1, 2]

def test_train_model(tmp_path):
    """
    Tests the train_model function by setting up a temporary MLflow tracking URI and experiment,
    creating a sample DataFrame, and verifying that the returned classifier, test features, and
    test labels have the expected types and properties.
    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest for storing test artifacts.
    Asserts:
        - The returned classifier has a 'predict' method.
        - The test features (x_test) are a pandas DataFrame.
        - The test labels (y_test) are a pandas Series.
        - The lengths of x_test and y_test are equal.
    """
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_test.db")
    mlflow.set_experiment("test_experiment")
    df = pd.DataFrame({
        "Survived": [0, 1, 0, 1],
        "Age": [22, 38, 26, 35],
        "Sex_Encoded": [0, 1, 0, 1],
        "Deck_Encoded": [8, 1, 8, 2],
        "Embarked_Encoded": [2, 0, 2, 1]
    })
    clf, x_test, y_test = train_model.fn(df)
    assert hasattr(clf, "predict")
    assert isinstance(x_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert len(x_test) == len(y_test)

def test_evaluate_model():
    """
    Test the evaluate_model function by training a RandomForestClassifier on a simple dataset
    and verifying that the returned accuracy is a float between 0.0 and 1.0.
    This test ensures:
    - The evaluate_model.fn function returns a float value.
    - The returned accuracy is within the valid range [0.0, 1.0].
    """
    x = pd.DataFrame({"a": [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])
    clf = RandomForestClassifier().fit(x, y)
    acc = evaluate_model.fn(clf, x, y)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

#### INTEGRATION TESTS FOR PIPELINE MODULES ####
def test_pipeline_integration(tmp_path):
    """
    Integration test for the entire machine learning pipeline.
    This test performs the following steps:
    1. Creates a small synthetic Titanic-like dataset and saves it as a CSV file.
    2. Runs the pipeline steps: data loading, preprocessing, model training, and evaluation.
    3. Asserts that the pipeline produces valid outputs:
        - The evaluation metric (accuracy) is a float between 0 and 1.
        - The trained model has a 'predict' method.
        - The test features and labels are of the correct types.
    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest for file operations.
    Raises:
        AssertionError: If any pipeline step fails or outputs are not as expected.
    """
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
    df_loaded = load_data.fn(str(file_path))
    df_processed = preprocess_data.fn(df_loaded)
    clf, x_test, y_test = train_model.fn(df_processed)
    acc = evaluate_model.fn(clf, x_test, y_test)

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    assert hasattr(clf, "predict")
    assert isinstance(x_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
