# 🛰️ Rock vs Mine Classification using Logistic Regression

This project implements a **binary classification model** to distinguish between **rocks** and **mines** using sonar signal data.  
It demonstrates a clean, modular, and reproducible machine learning workflow suitable for portfolio and production-style projects.

---

## 📊 Dataset

- **Name:** Sonar Mines vs Rocks Dataset  
- **Source:** Kaggle (originally from the UCI Machine Learning Repository)
- **Description:**  
  The dataset contains **60 continuous features** representing sonar signal energy at different frequencies.  
  Each observation is labeled as:
  - `R` → Rock
  - `M` → Mine

The dataset is automatically downloaded using `kagglehub`, ensuring reproducibility without manual file handling.

---

## 🧠 Approach

1. **Data Loading**
   - Dataset downloaded programmatically using `kagglehub`
   - Clean separation of data access logic

2. **Exploratory Data Analysis**
   - Missing value checks
   - Descriptive statistics
   - Class distribution analysis

3. **Preprocessing**
   - Feature/label separation
   - Feature scaling using `StandardScaler` (required for Logistic Regression)

4. **Modeling**
   - Logistic Regression classifier
   - Stratified train/test split
   - Proper solver and iteration configuration

5. **Evaluation**
   - Accuracy score
   - Full classification report (precision, recall, F1-score)

6. **Prediction**
   - Utility function for predicting new sonar readings


---

## 🚀 How to Run the Project

### 1️⃣ Create and activate a virtual environment
```cmd
python -m venv venv
venv\Scripts\activate.bat

