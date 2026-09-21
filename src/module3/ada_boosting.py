import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# ADA BOOSTING CLASSIFICATION
# Dataset: Placement Prediction 50K dataset
# ============================================================

DATA_PATH = r"C:C:\Users\psais\PycharmProjects\PlacementPrediction\Data\placement_predict_50k Dataset (2).csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# -------------------- Load dataset --------------------
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("                 ADA BOOSTING")
print("=" * 60)
print(f"Dataset shape: {df.shape}")

# -------------------- Target --------------------
TARGET = "PlacementStatus"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found.")

# -------------------- Select numeric features --------------------
exclude_cols = [
    TARGET,
    "StudentID",
    "IsAnomaly",
    "Salary Package"
]

feature_cols = [
    col for col in df.select_dtypes(include=np.number).columns
    if col not in exclude_cols
]

X = df[feature_cols].copy()
y = df[TARGET].copy()

print(f"\nFeatures used ({len(feature_cols)}):")
print(feature_cols)

# -------------------- Train/Test split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# -------------------- Missing values --------------------
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# -------------------- Scaling --------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------- AdaBoost model --------------------
base_model = DecisionTreeClassifier(
    max_depth=1,
    random_state=RANDOM_STATE
)

model = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=100,
    learning_rate=0.5,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

# -------------------- Prediction --------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTraining samples: {len(y_train):,}")
print(f"Testing samples : {len(y_test):,}")
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------- Confusion matrix --------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()
plt.title("AdaBoost - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "adaboost_confusion_matrix.png"
    ),
    dpi=300
)
plt.close()

# -------------------- Feature importance --------------------
importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\nTop Feature Importances:")
print(importance)

plt.figure(figsize=(9, 6))
importance.sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("AdaBoost Feature Importance")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "adaboost_feature_importance.png"
    ),
    dpi=300
)
plt.close()

# -------------------- Save report --------------------
with open(
    os.path.join(
        OUTPUT_DIR,
        "adaboost_report.txt"
    ),
    "w"
) as f:
    f.write("ADABOOST CLASSIFICATION REPORT\n")
    f.write("=" * 50 + "\n")
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Training samples: {len(y_train)}\n")
    f.write(f"Testing samples: {len(y_test)}\n")
    f.write(f"Features used: {len(feature_cols)}\n")
    f.write(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nFeature Importance:\n")
    f.write(importance.to_string())

print("\nOutput files saved in:", OUTPUT_DIR)
print("- adaboost_confusion_matrix.png")
print("- adaboost_feature_importance.png")
print("- adaboost_report.txt")
print("\nAdaBoost completed successfully.")