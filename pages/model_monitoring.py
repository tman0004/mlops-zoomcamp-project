import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
from scipy.stats import ks_2samp
from pipeline import preprocess_data

import matplotlib.pyplot as plt

st.title("Model Monitoring Dashboard")

ref_data = pd.read_csv('data/train.csv')
cur_data = pd.read_csv('data/test.csv')

# Preprocess the data
ref_data = preprocess_data(ref_data).drop(columns=['Survived'], errors='ignore')  # Drop target column if exists
cur_data = preprocess_data(cur_data)

st.header("Data Overview")
st.write("Reference Data Shape:", ref_data.shape)
st.write("Current Data Shape:", cur_data.shape)

st.header("Feature Drift Detection")
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

# write code to to show prediction distribution
st.header("Prediction Distribution")
mlflow.set_tracking_uri('sqlite:///mlflow.db')
model = mlflow.pyfunc.load_model(model_uri="models:/titanic_model/1")
predictions = model.predict(cur_data)
fig_pred, ax_pred = plt.subplots()
ax_pred.hist(predictions, bins=30, alpha=0.7, color='skyblue')
ax_pred.set_title("Prediction Distribution")
ax_pred.set_xlabel("Predicted Value")
ax_pred.set_ylabel("Frequency")
st.pyplot(fig_pred)