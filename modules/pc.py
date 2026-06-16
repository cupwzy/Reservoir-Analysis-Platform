import numpy as np
import pandas as pd


# ===============================
# ✅ PC数据准备（完全独立🔥）
# ===============================
def prepare_pc_data(df):

    df = df.copy()

    # 数值转换
    for col in ["PTR_P", "PORE_V_P"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 清洗
    df = df[
        (df["PTR_P"] > 0) &
        (df["PORE_V_P"] > 0)
    ]

    # log_PTR
    df["log_PTR"] = np.log10(df["PTR_P"])

    return df


# ===============================
# ✅ Pc曲线生成
# ===============================
def generate_pc_curve(df_cls):

    df_cls = df_cls.sort_values("log_PTR")

    # 分箱
    bins = np.linspace(
        df_cls["log_PTR"].min(),
        df_cls["log_PTR"].max(),
        50
    )

    df_cls["bin"] = np.digitize(df_cls["log_PTR"], bins)

    grouped = df_cls.groupby("bin").agg({
        "log_PTR": "mean",
        "PORE_V_P": "mean"
    }).dropna()

    grouped = grouped.rolling(3, center=True).mean().dropna()

    return grouped