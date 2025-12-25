from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd




def train_model(df: pd.DataFrame):
    X = df.drop(columns=["quality"])
    y = df["quality"]


    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
    )


    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)


    preds = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))


    return model, X