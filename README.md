# MLOps Zoomcamp 2025 Final Project

This project was created for the MLOps Zoomcamp final project. The goal of the project was to create an end-to-end ML project that implements proper MLOps techniques. Because this project and course was focused on the implementation of MLOps and best practices, I went with a simple dataset and model and focused on the MLOps side of the project. I decided to use the Titanic dataset and train a model to predict whether a passanger of the Titanic would survive or not.

## Technologies Used
- AWS - Used S3 to store model artifacts from MLflow. Used DynamoDB to store model predictions from web app which gets read in by the model monitoring dashboard.
- MLflow - Used for experiment tracking and model registry that links to S3.
- Prefect - Used for workflow orchestration and is fully deployed into web app.
- Streamlit - Deployed using Streamlit Cloud Community which hosts the web applciation that allows users to create predictions and view live results from the model monitoring dashboard.
- Pytest - Used for unit and integration tests
- Flake8 and Pylint - Used for code formatting and linting.
- Pre-commit - Used to create pre-commit hooks for git.
- GitHub Actions - Used to implement CI/CD pipeline.

## Usage
I've created a web application that allows users to input predictions as well as view model results and metrics from the model monitoring dashboard on the web app. 

Here is the [link](https://mlops-zoomcamp-project-ypwqjp8kfdaywqmf6svxyc.streamlit.app/) to the streamlit web app.

![web app screenshot 1](https://github.com/tman0004/mlops-zoomcamp-project/blob/main/imgs/img1.png)

![web app screenshot 2](https://github.com/tman0004/mlops-zoomcamp-project/blob/main/imgs/img2.png)

## Local Installation
**For project reviewers**: Because this project was developed with AWS, it's not very straightforward to run this project locally for others. I've deployed the web app to Streamlit Cloud Community because of this so that it is easy to see the full project (and showcase the deployment aspect). If you'd like to attempt to run it locally, you will need to create your own AWS account and S3 bucket and link the model registry to that bucket. You will then need to create your own DynamoDB to store model predictions and view model monitoring results. Once that is created, create a `.streamlit` folder and store your AWS credentials in a `secrets.toml` file. 

All of the required packages are in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

[Link](https://www.kaggle.com/datasets/heptapod/titanic) to the Titanic dataset.
