import numpy as np
import pandas as pd




def find_correlated(df: pd.DataFrame, label: str, corr_thresh: float, label_thresh: float):
    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    collinear = [c for c in upper.columns if any(upper[c] > corr_thresh)]
    weak_label = corr_matrix[label].drop(label)
    weak_label = weak_label[weak_label < label_thresh].index.tolist()
    return collinear, weak_label




def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop, weak = find_correlated(df, "quality", 0.8, 0.05)
    df = df.drop(columns=cols_to_drop + weak)
    df["quality"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)
    return df