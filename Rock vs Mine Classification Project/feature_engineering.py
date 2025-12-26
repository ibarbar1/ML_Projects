from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess(df: pd.DataFrame):
    X = df.drop(columns=["label"])
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler