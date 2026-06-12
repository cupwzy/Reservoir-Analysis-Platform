import pandas as pd

def load_master_table(file):
    df = pd.read_excel(file)

    required_cols = ["Name", "Well symbol", "Surface X", "Surface Y", "Target"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df