import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(r"C:\Users\psais\Downloads\placement_predict_50k Dataset (2).csv")

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


features = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore",
    "AttendancePercent"
]

target = "PlacementStatus"

X = df[features]
y = df[target]


# ============================================================
# 4. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

# Median is calculated using TRAINING data only.
# This prevents validation-data leakage.

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)


print("\nMissing values handled using median imputation.")


# ============================================================
# 6. FUNCTION FOR LOGISTIC REGRESSION
# ============================================================

def evaluate_model(X_train, X_val, y_train, y_val, name):

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Validation Accuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    print(classification_report(y_val, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, predictions))

    return model, accuracy


# ============================================================
# 7. UNSCALED LOGISTIC REGRESSION
# ============================================================

model_unscaled, acc_unscaled = evaluate_model(
    X_train_imputed,
    X_val_imputed,
    y_train,
    y_val,
    "Logistic Regression - Unscaled"
)


# ============================================================
# 8. STANDARD SCALER
# ============================================================

standard_scaler = StandardScaler()

X_train_standard = standard_scaler.fit_transform(
    X_train_imputed
)

X_val_standard = standard_scaler.transform(
    X_val_imputed
)


model_standard, acc_standard = evaluate_model(
    X_train_standard,
    X_val_standard,
    y_train,
    y_val,
    "Logistic Regression - StandardScaler"
)


# ============================================================
# 9. MIN-MAX SCALER
# ============================================================

minmax_scaler = MinMaxScaler()

X_train_minmax = minmax_scaler.fit_transform(
    X_train_imputed
)

X_val_minmax = minmax_scaler.transform(
    X_val_imputed
)


model_minmax, acc_minmax = evaluate_model(
    X_train_minmax,
    X_val_minmax,
    y_train,
    y_val,
    "Logistic Regression - MinMaxScaler"
)


# ============================================================
# 10. COMPARE RESULTS
# ============================================================

results = {
    "Unscaled": acc_unscaled,
    "StandardScaler": acc_standard,
    "MinMaxScaler": acc_minmax
}


print("\n\nFINAL COMPARISON")
print("=" * 60)

for name, accuracy in results.items():

    print(
        f"{name:<20} : "
        f"{accuracy:.4f}"
    )


# ============================================================
# 11. PLOT ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    results.keys(),
    results.values()
)

plt.xlabel("Scaling Method")
plt.ylabel("Validation Accuracy")
plt.title("Logistic Regression - Scaling Comparison")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig("logistic_scaler_comparison.png")

plt.show()


# ============================================================
# 12. FEATURE COEFFICIENTS
# ============================================================

print("\nFEATURE COEFFICIENTS")
print("=" * 60)

for feature, coefficient in zip(
    features,
    model_standard.coef_[0]
):

    print(
        f"{feature:<25} "
        f"{coefficient:.4f}"
    )


# ============================================================
# 13. PREDICT A NEW STUDENT
# ============================================================

new_student = pd.DataFrame({

    "CGPA": [8.5],

    "AptitudeTestScore": [82],

    "CodingTestScore": [85],

    "MockInterviewScore": [78],

    "AttendancePercent": [90]

})


# First handle missing values
new_student_imputed = imputer.transform(
    new_student
)
# Then use the SAME StandardScaler
new_student_scaled = standard_scaler.transform(
    new_student_imputed
)
# Prediction
prediction = model_standard.predict(
    new_student_scaled
)
# Probability
probability = model_standard.predict_proba(
    new_student_scaled
)
print("\nNEW STUDENT PREDICTION")
print("=" * 60)
if prediction[0] == 1:
    print("Prediction: PLACED")
else:
    print("Prediction: NOT PLACED")
print(
    "Probability of Not Placed:",
    round(probability[0][0], 4)
)
print(
    "Probability of Placed:",
    round(probability[0][1], 4)
)