import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




def explore(df: pd.DataFrame) -> None:
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nMissing values:\n", df.isnull().sum())
    print("\nTypes:\n", df.dtypes)
    print("\nStats:\n", df.describe())
    print("\nClass distribution:\n", df['quality'].value_counts())




def plot_correlations(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()