import pandas as pd
import numpy as np
import os
import joblib
import optuna
import optuna.visualization as vis

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

def objective(trial, X_train, y_train, X_val, y_val):

    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 150),
        max_depth=trial.suggest_int("max_depth", 5, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    return accuracy_score(y_val, y_val_pred)

def train_model():

    # ===============================
    # 1. 读取数据
    # ===============================
    file_path = "data/training_dataset.xlsx"

    if not os.path.exists(file_path):
        raise FileNotFoundError("❌ training_dataset.xlsx not found in /data")

    df = pd.read_excel(file_path, skiprows=[1])

    print(f"✅ Raw data: {df.shape}")

    # ===============================
    # 2. 标签处理
    # ===============================
    df = df[pd.to_numeric(df["Labelling260205"], errors="coerce").notna()]
    df["Label"] = df["Labelling260205"].astype(int)

    df = df[df["Label"].between(1, 8)]

    print(f"✅ After label clean: {df.shape}")

    # ===============================
    # 3. 特征选择
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
        raise ValueError(f"Missing columns: {missing_cols}")

    df = df.dropna(subset=feature_cols)

    X = df[feature_cols]
    y = df["Label"]

    print(f"✅ Final dataset: {X.shape}")

    # ===============================
    # 4. 数据拆分（70 / 15 / 15）
    # ===============================
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    print("\n📊 Dataset Split:")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    # ===============================
    # 5. 模型训练
    # ===============================
    
    print("\n🚀 Running Optuna Optimization...")

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=30
    )
    
    print("\n📊 Generating Optuna plots...")

    # 1. Optimization History
    fig1 = vis.plot_optimization_history(study)
    fig1.show()

    # 2. 参数重要性
    fig2 = vis.plot_param_importances(study)
    fig2.show()

    # 3. 参数关系
    fig3 = vis.plot_parallel_coordinate(study)
    fig3.show()

    print("\n✅ Best Parameters:", study.best_params)
    print("✅ Best Validation Score:", study.best_value)
    
    #用最优参数重新训练模型
    best_params = study.best_params
    
    model = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)



    # ===============================
    # 6. 验证集评估（调参用）
    # ===============================
    print("\n📊 Validation Evaluation:")

    y_val_pred = model.predict(X_val)

    print(classification_report(y_val, y_val_pred))

    # ===============================
    # 7. 测试集评估（最终结果）
    # ===============================
    print("\n📊 Test Evaluation:")

    y_test_pred = model.predict(X_test)

    print(classification_report(y_test, y_test_pred))

    # ===============================
    # 8. 混淆矩阵
    # ===============================
    print("\n📊 Confusion Matrix (Test):")

    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    # ===============================
    # 9. 保存模型
    # ===============================
    os.makedirs("models", exist_ok=True)

    model_version = "v1"
    model_path = f"models/rf_model_{model_version}.pkl"

    joblib.dump(model, model_path)

    # 更新 latest
    joblib.dump(model, "models/rf_model_latest.pkl")

    print(f"\n✅ Model saved: {model_path}")

    return model


# ===============================
# 主入口
# ===============================
if __name__ == "__main__":

    train_model()
