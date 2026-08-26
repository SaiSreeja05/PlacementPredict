import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = {
    "CGPA": [8.5, 7.2, 9.0, 6.8, 8.0, 7.0, 8.7, 7.5],
    "AptitudeScore": [85, 60, 90, 55, 75, 65, 82, 70],
    "Internship": [1, 0, 2, 0, 1, 0, 2, 0],
    "Placement": [1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
print("\n==============================")
print("        DATASET")
print("==============================")
print(df)

# Features (input variables)
X = df[["CGPA", "AptitudeScore", "Internship"]]

# Target (what we want to predict)
y = df["Placement"]
print("\n==============================")
print("        FEATURES (X)")
print("==============================")
print(X)

print("\n==============================")
print("        TARGET (y)")
print("==============================")
print(y)
# =========================================================
# 3. SPLIT DATA INTO TRAINING AND TESTING DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\n==============================")
print("      TRAINING DATA")
print("==============================")
print(X_train)

print("\nTraining target:")
print(y_train)

print("\n==============================")
print("       TESTING DATA")
print("==============================")
print(X_test)

print("\nTesting target:")
print(y_test)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("\n==============================")
print("    DECISION TREE TRAINED")
print("==============================")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\nTraining predictions:")
print(y_train_pred)

print("\nTesting predictions:")
print(y_test_pred)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n==============================")
print("         ACCURACY")
print("==============================")

print("Training Accuracy:", round(train_accuracy, 4))
print("Testing Accuracy :", round(test_accuracy, 4))

print("\n==============================")
print("       TREE INFORMATION")
print("==============================")

print("Tree Depth:", model.get_depth())
print("Number of Leaves:", model.get_n_leaves())

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

print("\n==============================")
print("      FEATURE IMPORTANCE")
print("==============================")

for feature, value in importance.items():
    print(f"{feature}: {value:.4f}")

plt.figure(figsize=(14, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Placed", "Placed"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree for Student Placement")

plt.tight_layout()

plt.savefig(
    "decision_tree.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(8, 5))

importance.sort_values().plot(
    kind="barh"
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Decision Tree Feature Importance")

plt.tight_layout()

plt.savefig(
    "decision_tree_feature_importance.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# New student's information:
#
# CGPA = 8.2
# Aptitude Score = 78
# Internship = 1

new_student = pd.DataFrame(
    [[8.2, 78, 1]],
    columns=["CGPA", "AptitudeScore", "Internship"]
)

prediction = model.predict(new_student)

print("\n==============================")
print("      NEW STUDENT")
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
print("       FINAL SUMMARY")
print("==============================")

print("Training Accuracy :", round(train_accuracy, 4))
print("Testing Accuracy  :", round(test_accuracy, 4))
print("Tree Depth         :", model.get_depth())
print("Number of Leaves   :", model.get_n_leaves())
print("\nMost Important Features:")
for feature, value in importance.items():
    print(f"  {feature:<20} {value:.4f}")
print("\nDecision Tree implementation completed successfully.")