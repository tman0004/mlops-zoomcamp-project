import streamlit as st
import pickle
import numpy as np
import mlflow.pyfunc
from pipeline import load_data, preprocess_data

mlflow.set_tracking_uri('sqlite:///mlflow.db')
model = mlflow.pyfunc.load_model(model_uri="models:/titanic_model/2")

st.title("Titanic Model Prediction")

df = load_data('data/test.csv')
st.write("Test Data:")
st.write(df)

if st.button("Predict"):
    # Preprocess the data
    preprocessed_df = preprocess_data(df)

    # Make predictions
    predictions = model.predict(preprocessed_df)

    df['Survived'] = predictions
    st.write("Test Data w/ Predictions:")
    st.write(df)