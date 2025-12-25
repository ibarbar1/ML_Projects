🍷 Wine Quality Classification (End-to-End ML Project)
📌 Project Overview

This project implements a complete, end-to-end machine learning pipeline to classify red wine quality as Good or Bad based on physicochemical properties.

The focus is not only on model accuracy, but on:

- clean project structure
- reproducibility
- modular, production-style Python code
- clear separation of concerns (data loading, EDA, preprocessing, modeling, inference)

🎯 Problem Statement

Given a set of physicochemical attributes of red wine (acidity, sulphates, alcohol, etc.), can we predict whether a wine is of good quality?

The original quality score is transformed into a binary classification problem:

1 → Good quality (quality ≥ 7)

0 → Bad quality (quality < 7)

📊 Dataset

Source: UCI Machine Learning Repository

Dataset: Red Wine Quality (Cortez et al., 2009)

Access: Automatically downloaded via kagglehub

Features include:

fixed acidity

volatile acidity

citric acid

chlorides

sulphates

alcohol

and more

Target variable: quality

🧱 Project Structure
Wine Quality Classification Project/
│
├── main.py                  # Entry point
├── data_loader.py           # Dataset download & loading
├── exploration.py           # EDA and visualization
├── feature_engineering.py   # Correlation handling & preprocessing
├── model.py                 # Training and evaluation
├── predictor.py             # Inference helper
│
├── requirements.txt
├── README.md
├── venv/                    # Virtual environment (ignored by Git)

⚙️ Environment Setup (Windows CMD)
1️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate.bat

2️⃣ Install dependencies
pip install -r requirements.txt

▶️ Running the Project

From the project root directory:

python main.py

🧪 What Happens When You Run main.py

Dataset is downloaded (cached on subsequent runs)

Exploratory Data Analysis is printed:

dataset shape

missing values

data types

summary statistics

class distribution

Correlation heatmap is displayed

⚠️ Execution pauses until the plot window is closed

Feature preprocessing

multicollinearity reduction

weak predictors removed

target converted to binary

Model training

RandomForestClassifier

stratified train/test split

Evaluation metrics printed

accuracy

precision / recall / F1-score

Sample prediction displayed

"Good Wine" or "Bad Wine"

🤖 Model Details

Algorithm: Random Forest Classifier

Reasoning:

handles non-linear relationships

robust to feature scaling

interpretable via feature importance

Typical test accuracy: ~80–87%

🔮 Example Prediction
from predictor import predict_quality

predict_quality(model, [
    7.4, 0.70, 0.00, 0.076,
    11.0, 34.0, 0.9978,
    3.51, 0.56, 9.4
])


Output:

"Good Wine"

🧠 Key Skills Demonstrated

Data exploration and validation

Feature selection via correlation analysis

Binary classification framing

Clean modular Python design

Reproducible environment setup

Model evaluation and inference

🚀 Possible Extensions

Save trained model using joblib

Add logging instead of print

Expose predictions via FastAPI

Add unit tests (pytest)

Add CI workflow (GitHub Actions)

Power BI integration via REST API

📜 License

MIT License