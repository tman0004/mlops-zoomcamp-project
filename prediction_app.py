"""
Titanic Model Prediction Streamlit App
This module provides a Streamlit web application for predicting the survival of Titanic passengers
using a machine learning model served via MLflow. Users can input passenger details through a form,
and the app will display the predicted survival outcome. The app also logs each prediction to an AWS
DynamoDB table for monitoring and analysis.
Features:
- Interactive form for user input of passenger features (class, gender, age, fare, etc.).
- Real-time prediction using a pre-trained MLflow model.
- Storage of prediction results and input features in DynamoDB for monitoring.
- Integration with a custom preprocessing pipeline for feature engineering.
- User interface includes a sidebar link to a model monitoring dashboard.
Dependencies:
- streamlit
- pandas
- numpy
- mlflow
- boto3
- decimal
- Custom modules: pipeline (with load_data, preprocess_data)
"""
from decimal import Decimal
import boto3
import streamlit as st
import pandas as pd
import mlflow.pyfunc
from pipeline import preprocess_data

mlflow.set_tracking_uri('sqlite:///mlflow.db')
model = mlflow.pyfunc.load_model(model_uri="models:/titanic_model/2")

st.title("Titanic Model Prediction")
st.write("""This app predicts whether a passenger survived the Titanic disaster based
         on various features. You can input the passenger details in the form below, 
         and the model will predict the survival status.""")
st.write("""You can also view the model monitoring dashboard from the left sidebar
         to see how the model is performing over time.""")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    col1, col2, col3 = st.columns([1, 2, 2])

    with col1:
        pclass = st.radio("Passenger Class", options=[1, 2, 3], index=0)
        sex = st.radio('Gender', options=['Male', 'Female'], index=0).lower()

    with col2:
        age = st.number_input("Age", min_value=0, max_value=100)
        fare = st.number_input("Fare", min_value=0.0)
        sibsp = st.number_input("Number of Siblings or Spouses on Board", min_value=0, max_value=10)

    with col3:
        parch = st.number_input("Number of Parents or Children on Board", min_value=0, max_value=10)
        cabin = st.selectbox(
            "Cabin Deck", options=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'], index=8)
        embarked = st.selectbox(
            "Embarked", options=['Cherbourg', 'Queenstown', 'Southampton'], index=0)[0]

    submitted = st.form_submit_button("Submit")

    if submitted:
        prediction_df = pd.DataFrame({
            'PassengerId': None,
            'Pclass': [pclass],
            'Name': None,
            'Sex': [sex],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Ticket': None,
            'Fare': [fare],
            'Cabin': [cabin],
            'Embarked': [embarked]
        })
        preprocessed_df = preprocess_data(prediction_df)
        predictions = model.predict(preprocessed_df)
        if predictions[0] == 1:
            st.success("The passenger is predicted to have survived.")
        else:
            st.error("The passenger is predicted not to have survived.")

        prediction_df['Survived'] = predictions

        dynamodb = boto3.resource('dynamodb', region_name='us-west-1')
        table = dynamodb.Table('titanic_predictions')
        response = table.scan()
        items = response['Items']
        if items:
            max_id = max(int(item['id']) for item in items)
        else:
            max_id = 0

        new_id = max_id + 1

        new_row = {
            "id": int(new_id),
            "Fare": Decimal(str(float(prediction_df['Fare'][0]))),
            "Deck_Encoded": int(prediction_df['Deck_Encoded'][0]),
            "Pclass": int(prediction_df['Pclass'][0]),
            "Embarked_Encoded": int(prediction_df['Embarked_Encoded'][0]),
            "Sex_Encoded": int(prediction_df['Sex_Encoded'][0]),
            "Parch": int(prediction_df['Parch'][0]),
            "SibSp": int(prediction_df['SibSp'][0]),
            "Age": Decimal(str(float(prediction_df['Age'][0]))),
            "Survived": int(prediction_df['Survived'][0])
        }

        table.put_item(Item=new_row)
