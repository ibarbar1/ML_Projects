import pandas as pd

def explore(df: pd.DataFrame) -> None:
    print("Missing values:\n", df.isnull().sum())
    print("\nData types:\n", df.dtypes)
    print("\nDescriptive statistics:\n", df.describe())
    print("\nClass distribution:\n", df["label"].value_counts())
