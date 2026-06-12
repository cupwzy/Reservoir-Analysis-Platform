import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_data(file_path):

    # ===============================
    # 1. 读取数据
    # ===============================
    df = pd.read_excel(file_path, skiprows=[1])

    print(f"✅ Raw data shape: {df.shape}")

    # ===============================
    # 2. 标签处理
    # ===============================
    df = df[pd.to_numeric(df["Labelling260205"], errors="coerce").notna()]
    df["Label"] = df["Labelling260205"].astype(int)

    df = df[df["Label"].between(1, 8)]

    print(f"✅ After label clean: {df.shape}")

    # ===============================
    # 3. 特征列（模型必须）
    # ===============================
    feature_cols = [
        "CKH_clean",
        "CPOR_clean",
        "PTR_P",
        "PC_STRESS_CORR",
        "SW_STRESS_CORR"
    ]

    # ✅ 可选列（用于网页展示）
    optional_cols = [
        "PORE_V_P",
        "wellName"
    ]

    # ===============================
    # 4. 检查字段
    # ===============================
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    # ===============================
    # 5. 清洗数据
    # ===============================
    df = df.dropna(subset=feature_cols)

    # ✅ 只保留需要的列（但保留完整）
    keep_cols = feature_cols + ["Label"]

    for col in optional_cols:
        if col in df.columns:
            keep_cols.append(col)

    df = df[keep_cols].copy()

    print(f"✅ Final dataset: {df.shape}")

    # ===============================
    # 6. 拆分数据（70 / 15 / 15）
    # ===============================
    X = df
    y = df["Label"]

    df_train, df_temp = train_test_split(
        X,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        random_state=42,
        stratify=df_temp["Label"]
    )

    # ===============================
    # 7. 输出统计
    # ===============================
    print("\n📊 Dataset Split:")
    print(f"Train: {df_train.shape}")
    print(f"Validation: {df_val.shape}")
    print(f"Test: {df_test.shape}")

    print("\n📊 Label Distribution:")

    print("\nTrain:")
    print(df_train["Label"].value_counts(normalize=True))

    print("\nValidation:")
    print(df_val["Label"].value_counts(normalize=True))

    print("\nTest:")
    print(df_test["Label"].value_counts(normalize=True))

    # ===============================
    # 8. 保存数据（网页可用 ✅）
    # ===============================
    os.makedirs("data_split", exist_ok=True)

    df_train.to_excel("data_split/train.xlsx", index=False)
    df_val.to_excel("data_split/val.xlsx", index=False)
    df_test.to_excel("data_split/test.xlsx", index=False)

    print("\n✅ Data saved:")
    print("data_split/train.xlsx")
    print("data_split/val.xlsx")
    print("data_split/test.xlsx")

    return df_train, df_val, df_test


# ===============================
# 主入口
# ===============================
if __name__ == "__main__":

    file_path = "data/training_dataset.xlsx"

    if not os.path.exists(file_path):
        raise FileNotFoundError("❌ 请把数据放在 data/training_dataset.xlsx")

    split_data(file_path)