import numpy as np
import pandas as pd


def train_rca_monotonic(df):

    df = df[
        (df["CKH_clean"] > 0) &
        (df["CPOR_clean"] > 0) &
        (df["CPOR_clean"] < 1)
    ].copy()

    # log空间
    df["logk"] = np.log10(df["CKH_clean"])
    df["phi"] = df["CPOR_clean"]

    # ===============================
    # Step1：整体拟合得到统一斜率
    # ===============================
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    X = df["phi"].values.reshape(-1, 1)
    y = df["logk"].values

    model.fit(X, y)
    b_global = model.coef_[0]

    # ===============================
    # Step2：每个Type计算截距
    # ===============================
    temp = []

    for t, g in df.groupby("PoreType"):

        a = (g["logk"] - b_global * g["phi"]).mean()

        temp.append((t, a))

    # ===============================
    # Step3：按a排序
    # ===============================
    temp = sorted(temp, key=lambda x: x[1])

    # ===============================
    # Step4：强制单调递增
    # ===============================
    models = {}
    prev_a = -np.inf

    for t, a in temp:

        # 强制不交叉
        a = max(a, prev_a + 0.05)

        models[t] = {
            "a": a,
            "b": b_global
        }

        prev_a = a

    return models
