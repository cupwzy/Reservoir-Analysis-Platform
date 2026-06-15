import numpy as np
from sklearn.linear_model import LinearRegression

def train_rca_model(df):

    df = df[
        (df["CKH_clean"] > 0) &
        (df["CPOR_clean"] > 0) &
        (df["CPOR_clean"] < 1)
    ]

    models = {}

    for t, group in df.groupby("PoreType"):

        phi = group["CPOR_clean"].values
        k = group["CKH_clean"].values

        X = np.log10(phi).reshape(-1, 1)
        y = np.log10(k)

        model = LinearRegression()
        model.fit(X, y)

        a = 10 ** model.intercept_
        b = model.coef_[0]

        models[t] = {"a": a, "b": b}

    return models


def predict_k(phi, pore_type, models):

    if pore_type not in models:
        return None

    a = models[pore_type]["a"]
    b = models[pore_type]["b"]

    return a * (phi ** b)