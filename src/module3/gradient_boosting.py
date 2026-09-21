import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = {
    "CGPA": [8.5, 7.2, 9.0, 6.8, 8.0, 7.0, 8.7, 7.5, 8.2, 6.5, 9.2, 7.8, 6.9, 8.9],
    "AptitudeScore": [85, 60, 90, 55, 75, 65, 82, 70, 78, 58, 92, 73, 62, 88],
    "Internship": [1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 1, 0, 2],
    "Placement": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

print("\n==============================")
print("       14-SAMPLE DATASET")
print("==============================")

print(df)

X = df[["CGPA", "AptitudeScore", "Internship"]]
y = df["Placement"]

print("\n==============================")
print("          FEATURES")
print("==============================")

print(X)

print("\n==============================")
print("           TARGET")
print("==============================")

print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\n==============================")
print("       TRAINING SAMPLES")
print("==============================")

print(len(X_train))

print("\n==============================")
print("        TESTING SAMPLES")
print("==============================")

print(len(X_test))

model = GradientBoostingClassifier(
    n_estimators=10,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("\n==============================")
print("   GRADIENT BOOSTING TRAINED")
print("==============================")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\nTraining Predictions:")
print(y_train_pred)

print("\nTesting Predictions:")
print(y_test_pred)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n==============================")
print("          ACCURACY")
print("==============================")

print("Training Accuracy:", round(train_accuracy, 4))
print("Testing Accuracy :", round(test_accuracy, 4))

print("\n==============================")
print("  GRADIENT BOOSTING INFORMATION")
print("==============================")

print("Number of Trees:", model.n_estimators)
print("Learning Rate:", model.learning_rate)
print("Maximum Tree Depth:", model.max_depth)

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

print("\n==============================")
print("      FEATURE IMPORTANCE")
print("==============================")

for feature, value in importance.items():
    print(f"{feature:<20} : {value:.4f}")

plt.figure(figsize=(8, 5))

importance.sort_values().plot(kind="barh")

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Gradient Boosting Feature Importance")

plt.tight_layout()

plt.savefig(
    "gradient_boosting_feature_importance.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

new_student = pd.DataFrame(
    [[8.2, 78, 1]],
    columns=["CGPA", "AptitudeScore", "Internship"]
)

prediction = model.predict(new_student)

print("\n==============================")
print("       NEW STUDENT")
print("==============================")

print(new_student)

print("\nPrediction:", prediction[0])

if prediction[0] == 1:
    print("Result: Placed")
else:
    print("Result: Not Placed")

probability = model.predict_proba(new_student)

print("\n==============================")
print("    PREDICTION PROBABILITY")
print("==============================")

print("Not Placed:", round(probability[0][0], 4))
print("Placed    :", round(probability[0][1], 4))

print("\n==============================")
print("      INDIVIDUAL TREES")
print("==============================")

for i, tree in enumerate(model.estimators_.ravel(), start=1):
    print(
        f"Tree {i}: "
        f"Depth = {tree.get_depth()}, "
        f"Leaves = {tree.get_n_leaves()}"
    )

print("\n==============================")
print("        FINAL SUMMARY")
print("==============================")

print("Number of samples :", len(df))
print("Number of trees   :", model.n_estimators)
print("Training Accuracy :", round(train_accuracy, 4))
print("Testing Accuracy  :", round(test_accuracy, 4))
print("Learning Rate     :", model.learning_rate)
print("Tree Depth Limit  :", model.max_depth)

print("\nGradient Boosting implementation completed successfully.")