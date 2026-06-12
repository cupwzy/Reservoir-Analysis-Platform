import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ===============================
# 主函数（用于云端调用）
# ===============================
def train_and_save_model():

    print("📌 Start training model...")

    # ===============================
    # 1. 读取数据
    # ===============================
    file_path = "data/training_dataset.xlsx"

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Training dataset not found. Please place file in data/training_dataset.xlsx"
        )

    df = pd.read_excel(file_path, skiprows=[1])

    print(f"✅ Data loaded: {df.shape}")

    # ===============================
    # 2. 清洗标签（只保留1–8）
    # ===============================
    df = df[pd.to_numeric(df["Labelling260205"], errors="coerce").notna()]

    df["Label"] = df["Labelling260205"].astype(int)

    df = df[df["Label"].between(1, 8)]

    print(f"✅ After label cleaning: {df.shape}")

    # ===============================
    # 3. 特征选择（必须和UI一致）
    # ===============================
    feature_cols = [
        "CKH_clean",
        "CPOR_clean",
        "PTR_P",
        "PC_STRESS_CORR",
        "SW_STRESS_CORR"
    ]

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    df = df.dropna(subset=feature_cols)

    X = df[feature_cols]
    y = df["Label"]

    print(f"✅ Final training data: {X.shape}")

    # ===============================
    # 4. 数据划分（用于评估）
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # ===============================
    # 5. 模型训练
    # ===============================
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1  
    )

    model.fit(X_train, y_train)

    print("✅ Model training finished")

    # ===============================
    # 6. 模型评估
    # ===============================
    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # ===============================
    # 7. 保存模型
    # ===============================
    os.makedirs("models", exist_ok=True)

    model_path = "models/rf_model.pkl"

    joblib.dump(model, model_path)

    print(f"\n✅ Model saved to: {model_path}")

    return model


# ===============================
# 允许直接运行（本地训练）
# ===============================
if __name__ == "__main__":

    train_and_save_model()
