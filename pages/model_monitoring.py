"""
Model Monitoring Dashboard for Titanic Predictions
This Streamlit application provides a dashboard for monitoring the performance and
 data drift of a deployed Titanic survival prediction model. It connects to an AWS DynamoDB 
 table to retrieve recent prediction data, compares it with reference training data, 
 and visualizes key performance indicators (KPIs) and feature distributions. 
 The dashboard includes:
- KPI metrics for total predictions and survival rates, with deltas from model launch.
- Data drift detection using the Kolmogorov-Smirnov (KS) test for selected features.
- Visual comparison of feature distributions between reference and current data.
- Integration with AWS DynamoDB for real-time prediction monitoring.
Dependencies:
    - streamlit
    - pandas
    - numpy
    - scipy
    - matplotlib
    - boto3
    - pipeline (custom preprocessing module)
"""
import boto3
import streamlit as st
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from pipeline import preprocess_data

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
ref_data = preprocess_data(ref_data).drop(columns=['Survived'], errors='ignore')

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
