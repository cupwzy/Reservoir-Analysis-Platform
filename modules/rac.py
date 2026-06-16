from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

__all__ = [
    "prepare_rca_data",
    "apply_fzi_typing",
    "train_rca_model"
]

def train_rca_model(df):

    df = df.copy()

    df["CKH_clean"] = pd.to_numeric(df["CKH_clean"], errors="coerce")
    df["CPOR_clean"] = pd.to_numeric(df["CPOR_clean"], errors="coerce")
    
    df = df[
            (df["CKH_clean"] > 0) &
            (df["CPOR_clean"] > 0) &
            (df["CPOR_clean"] < 1)
        ]

    df["log_k"] = np.log10(df["CKH_clean"])

    if len(df) < 10:
            raise ValueError("Not enough valid RCA data after cleaning")

    # ===============================
    # 模型
    # ==============================

    model = LinearRegression()
    model.fit(df[["CPOR_clean"]], df["log_k"])

    b_global = model.coef_[0]

    models = {}
    prev_a = -np.inf

    temp = []

    for t, group in df.groupby("PoreType"):

        if len(group) < 5:
            continue

        a = (group["log_k"] - b_global * group["CPOR_clean"]).mean()
        temp.append((t, a))

    temp = sorted(temp, key=lambda x: x[1])

    for t, a in temp:
        a = max(a, prev_a + 0.05)
        models[t] = {"a": a, "b": b_global}
        prev_a = a

    return models

def prepare_rca_data(df):

    df = df.copy()

    for col in ["CKH_clean", "CPOR_clean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        (df["CKH_clean"] > 0) &
        (df["CPOR_clean"] > 0) &
        (df["CPOR_clean"] < 1)
    ]

    return df
def apply_fzi_typing(df):
    df = df.copy()

    df["FZI"] = 0.0314 * np.sqrt(df["CKH_clean"] / df["CPOR_clean"]) / (1 - df["CPOR_clean"])

    bins = [0, 0.35, 0.8, 1.77, 4, 10]
    labels = [1, 2, 3, 4, 5]
    df["RF_PoreType"] = df["PoreType"]
    df["FZI_Type"] = pd.cut(df["FZI"], bins=bins, labels=labels)

    df = df.dropna(subset=["FZI_Type"])
    df["FZI_Type"] = df["FZI_Type"].astype(int)

    return df