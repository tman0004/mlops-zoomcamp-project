import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
from scipy.stats import ks_2samp
from pipeline import preprocess_data
import boto3

import matplotlib.pyplot as plt

st.title("Model Monitoring Dashboard")

ref_data = pd.read_csv('data/train.csv')
dynamodb = boto3.resource('dynamodb', region_name='us-west-1')
table = dynamodb.Table('titanic_predictions')
response = table.scan()
items = response['Items']

# If the table is large, handle pagination
while 'LastEvaluatedKey' in response:
    response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
    items.extend(response['Items'])

# Load into pandas DataFrame
cur_data = pd.DataFrame(items).drop(columns=['id'], errors='ignore')

# Preprocess the data
ref_data = preprocess_data(ref_data).drop(columns=['Survived'], errors='ignore')  # Drop target column if exists
# cur_data = preprocess_data(cur_data)

st.header("Data KPI Overview")
# Starting test dataset
previous_total = 418
previous_survival_pct = 37.96
previous_not_survival_pct = 62.04
pred_counts = cur_data['Survived'].value_counts(normalize=True) * 100
current_total = len(cur_data)
current_survive_pct = pred_counts.get(1, 0)
current_not_survive_pct = pred_counts.get(0, 0)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="Total Predictions",
        value=current_total,
        delta=current_total - previous_total,
        border=True,
        help="Delta values are from model launch"
    )
with col2:
    st.metric(
        label="Predicted to survive",
        value=f"{current_survive_pct:.2f}%",
        delta=f"{current_survive_pct - previous_survival_pct:.2f}%",
        border=True,
        help="Delta values are from model launch"
    )
with col3:
    st.metric(
        label="Predicted not to survive",
        value=f"{current_not_survive_pct:.2f}%",
        delta=f"{current_not_survive_pct - previous_not_survival_pct:.2f}%",
        border=True,
        help="Delta values are from model launch"
    )

st.header("Data Drift Detection")
feature = st.selectbox("Select Feature to Analyze", ref_data.columns)

fig, ax = plt.subplots()
ax.hist(ref_data[feature], bins=30, alpha=0.5, label='Reference')
ax.hist(cur_data[feature], bins=30, alpha=0.5, label='Current')
ax.legend()
ax.set_title(f"Distribution of {feature}")
st.pyplot(fig)

ks_stat, p_value = ks_2samp(ref_data[feature].dropna(), cur_data[feature].dropna())
st.write(f"KS Statistic: {ks_stat:.4f}")
st.write(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    st.error("Significant drift detected!")
else:
    st.success("No significant drift detected.")