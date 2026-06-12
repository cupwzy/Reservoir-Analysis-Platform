import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# 1. 读取数据
# ===============================
df = pd.read_excel("data/training_dataset.xlsx", skiprows=[1])

# ===============================
# 2. 清洗标签
# ===============================
df = df[pd.to_numeric(df["Labelling260205"], errors="coerce").notna()]
df["Label"] = df["Labelling260205"].astype(int)

df = df[df["Label"].between(1, 8)]

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

df = df.dropna(subset=feature_cols)

X = df[feature_cols]
y = df["Label"]

# ===============================
# 4. 数据划分
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5. 模型训练
# ===============================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 6. 评估
# ===============================
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 7. 混淆矩阵
# ===============================
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()

# ===============================
# 8. 特征重要性
# ===============================
importance = pd.Series(model.feature_importances_, index=feature_cols)

plt.figure(figsize=(6,5))
importance.sort_values().plot(kind="barh")
plt.title("Feature Importance")
plt.tight_layout()

plt.savefig("outputs/feature_importance.png", dpi=300)
plt.close()

# ===============================
# 9. 保存模型
# ===============================
joblib.dump(model, "models/rf_model.pkl")

print("\n✅ Model saved.")