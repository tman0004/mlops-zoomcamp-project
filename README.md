# MLOps Zoomcamp 2025 Final Project

This project was created for the MLOps Zoomcamp final project. The goal of the project was to create an end-to-end ML project that implements proper MLOps techniques. Because this project and course was focused on the implementation of MLOps and best practices, I went with a simple dataset and model and focused on the MLOps side of the project. I decided to use the Titanic dataset and train a model to predict whether a passanger of the Titanic would survive or not.

- **AWS**
  - **S3**: Stores model artifacts tracked by MLflow.
  - **DynamoDB**: Stores prediction records from the web app, which are visualized in the monitoring dashboard.
- **MLflow**: Handles experiment tracking and model registry, with S3 as the backend store.
- **Prefect**: Workflow orchestration used in the model pipeline and is deployed to trigger from the Streamlit app.
- **Streamlit**: Hosts the interactive web application (deployed via Streamlit Community Cloud).
- **Pytest**: Used for unit and integration testing.
- **Flake8 & Pylint**: Used for code style checking and linting.
- **Pre-commit**: Automates linting, formatting, and code checks on commits.
- **GitHub Actions**: Powers the CI/CD pipeline for automated testing and linting.

## 🚀 Web App

The deployed web application allows users to:
- Make predictions using a trained model.
- View live model performance and prediction data in a monitoring dashboard.

🔗 [**Live app**](https://mlops-zoomcamp-project-ypwqjp8kfdaywqmf6svxyc.streamlit.app/)

![Web app screenshot 1](https://github.com/tman0004/mlops-zoomcamp-project/blob/main/imgs/img1.png)
![Web app screenshot 2](https://github.com/tman0004/mlops-zoomcamp-project/blob/main/imgs/img2.png)

## 🧪 Local Installation

> ⚠️ **Note for project reviewers**: This project uses cloud services (AWS S3 and DynamoDB), so running it locally requires cloud credentials and setup. For convenience, the web app is deployed on Streamlit Cloud to demonstrate full functionality.

If you wish to run locally:

1. Set up an AWS account.
2. Create:
   - An S3 bucket for storing model artifacts.
   - A DynamoDB table for storing predictions.
3. Inside your project root, create a `.streamlit/secrets.toml` file and add your AWS credentials:

```toml
AWS_ACCESS_KEY_ID = "your-access-key"
AWS_SECRET_ACCESS_KEY = "your-secret-key"
AWS_DEFAULT_REGION = "your-region"
```

[Link](https://www.kaggle.com/datasets/heptapod/titanic) to the Titanic dataset.
