import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# 1. 读取数据（跳过单位行）
# ===============================
df = pd.read_excel("data/training_dataset.xlsx", skiprows=[1])

# ===============================
# 2. 选择特征 & 标签（根据你的表结构）
# ===============================
feature_cols = [
    "CKH",
    "CKH_clean",
    "CPOR",
    "CPOR_clean",
    "PTR_P",
    "PC_LAB",
    "SW_STRESS_CORR"
]

X = df[feature_cols]
y = df["Labelling260205"]   # ✅ 选择一个标签

# ===============================
# 3. 数据划分
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4. 模型训练
# ===============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 5. 预测
# ===============================
y_pred = model.predict(X_test)

print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# ✅ 6. 混淆矩阵
# ===============================
plt.figure(figsize=(6,5))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.show()

# ===============================
# ✅ 7. 特征重要性（最关键）
# ===============================
importance = pd.Series(model.feature_importances_, index=feature_cols)
importance = importance.sort_values()

plt.figure(figsize=(6,5))

importance.plot(kind="barh")

plt.title("Feature Importance")
plt.xlabel("Importance")

plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=300)
plt.show()

print("\n✅ Feature Importance:\n")
print(importance.sort_values(ascending=False))

# ===============================
# ✅ 8. 学习曲线（判断过拟合）
# ===============================
train_sizes, train_scores, val_scores = learning_curve(
    model,
    X, y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(6,5))

plt.plot(train_sizes, train_mean, marker='o', label="Training Score")
plt.plot(train_sizes, val_mean, marker='o', label="Validation Score")

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()

plt.grid()

plt.tight_layout()
plt.savefig("outputs/learning_curve.png", dpi=300)
plt.show()

# ===============================
# ✅ 9. 保存模型
# ===============================
joblib.dump(model, "models/rf_model.pkl")

print("\n✅ Model saved: models/rf_model.pkl")