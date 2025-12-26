import pandas as pd
import kagglehub
from pathlib import Path

def load_sonar_data() -> pd.DataFrame:
    """Download and load the Sonar dataset."""
    path = kagglehub.dataset_download("mattcarter865/mines-vs-rocks")
    dataset_path = Path(path) / "sonar.all-data.csv"
    df = pd.read_csv(dataset_path, header=None)

    # 60 feature columns + 1 label column
    df.columns = [f"feature_{i}" for i in range(60)] + ["label"]

    return df