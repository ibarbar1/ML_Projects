import pandas as pd
import kagglehub
from pathlib import Path




def load_wine_data() -> pd.DataFrame:
    """Download and load the red wine quality dataset."""
    path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
    dataset_path = Path(path) / "winequality-red.csv"
    return pd.read_csv(dataset_path)