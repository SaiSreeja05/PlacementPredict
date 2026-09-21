import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_PATH = r"C:\Users\psais\PycharmProjects\PlacementPrediction\Data\placement_predict_50k Dataset (2).csv"
OUTPUT_DIR = r"C:\Users\psais\PycharmProjects\PlacementPrediction\src\module4\pca"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n====================================")
print("        PCA ANALYSIS")
print("====================================")
print("\nLoading dataset...")
print(DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("\nDataset loaded successfully!")
print("Shape:", df.shape)

# ============================================================
# 3. SELECT NUMERIC FEATURES
# ============================================================

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

print("\nNumeric columns:")
print(numeric_columns)

# Columns that should NOT be used for PCA
columns_to_remove = [
    "StudentID",
    "PlacementStatus",
    "Salary Package",
    "IsAnomaly"
]

feature_columns = [
    col for col in numeric_columns
    if col not in columns_to_remove
]

print("\nFeatures used for PCA:")
print(feature_columns)

if len(feature_columns) == 0:
    raise ValueError("No numeric features available for PCA.")


X = df[feature_columns].copy()


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

print("\nChecking missing values...")

print(X.isnull().sum())

imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X)

print("\nMissing values handled using median imputation.")


# ============================================================
# 5. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

print("Features standardized successfully.")


# ============================================================
# 6. PCA
# ============================================================

print("\nRunning PCA...")

pca = PCA()

X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance)


# ============================================================
# 7. PRINT VARIANCE
# ============================================================

print("\n====================================")
print("   EXPLAINED VARIANCE")
print("====================================")

for i in range(min(10, len(explained_variance))):

    print(
        f"PC{i + 1}: "
        f"{explained_variance[i] * 100:.2f}%"
    )


# ============================================================
# 8. COMPONENTS REQUIRED
# ============================================================

def components_for_variance(threshold):

    return np.argmax(
        cumulative_variance >= threshold
    ) + 1


pc_80 = components_for_variance(0.80)
pc_90 = components_for_variance(0.90)
pc_95 = components_for_variance(0.95)

print("\n====================================")
print("   COMPONENT REQUIREMENTS")
print("====================================")

print("Components for 80% variance:", pc_80)
print("Components for 90% variance:", pc_90)
print("Components for 95% variance:", pc_95)


# ============================================================
# 9. SCREE PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance * 100,
    marker="o"
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")
plt.title("PCA Scree Plot")

plt.grid(True)

plt.tight_layout()

scree_path = os.path.join(
    OUTPUT_DIR,
    "scree_plot.png"
)

plt.savefig(scree_path, dpi=300)

plt.close()

print("\nSaved:")
print(scree_path)


# ============================================================
# 10. CUMULATIVE VARIANCE PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance * 100,
    marker="o"
)

plt.axhline(
    80,
    linestyle="--",
    label="80%"
)

plt.axhline(
    90,
    linestyle="--",
    label="90%"
)

plt.axhline(
    95,
    linestyle="--",
    label="95%"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")

plt.title("PCA Cumulative Explained Variance")

plt.legend()

plt.grid(True)

plt.tight_layout()

cumulative_path = os.path.join(
    OUTPUT_DIR,
    "cumulative_variance.png"
)

plt.savefig(cumulative_path, dpi=300)

plt.close()

print("Saved:")
print(cumulative_path)


# ============================================================
# 11. PCA LOADINGS
# ============================================================

loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_columns,
    columns=[
        f"PC{i + 1}"
        for i in range(len(feature_columns))
    ]
)


# ============================================================
# 12. TOP PC1 FEATURES
# ============================================================

print("\n====================================")
print("   TOP PC1 FEATURES")
print("====================================")

pc1_features = (
    loadings["PC1"]
    .abs()
    .sort_values(ascending=False)
    .head(10)
)

print(pc1_features)


# ============================================================
# 13. TOP PC2 FEATURES
# ============================================================

print("\n====================================")
print("   TOP PC2 FEATURES")
print("====================================")

if "PC2" in loadings.columns:

    pc2_features = (
        loadings["PC2"]
        .abs()
        .sort_values(ascending=False)
        .head(10)
    )

    print(pc2_features)


# ============================================================
# 14. SAVE LOADINGS
# ============================================================

loadings_path = os.path.join(
    OUTPUT_DIR,
    "pca_loadings.csv"
)

loadings.to_csv(loadings_path)

print("\nSaved:")
print(loadings_path)


# ============================================================
# 15. PC1 vs PC2 LOADINGS PLOT
# ============================================================

if len(feature_columns) >= 2:

    plt.figure(figsize=(10, 7))

    plt.scatter(
        loadings["PC1"],
        loadings["PC2"]
    )

    for feature in feature_columns:

        plt.annotate(
            feature,
            (
                loadings.loc[feature, "PC1"],
                loadings.loc[feature, "PC2"]
            ),
            fontsize=8
        )

    plt.xlabel("PC1 Loading")
    plt.ylabel("PC2 Loading")

    plt.title("PCA Feature Loadings")

    plt.axhline(0, linestyle="--")
    plt.axvline(0, linestyle="--")

    plt.grid(True)

    plt.tight_layout()

    loading_plot_path = os.path.join(
        OUTPUT_DIR,
        "loadings_biplot.png"
    )

    plt.savefig(
        loading_plot_path,
        dpi=300
    )

    plt.close()

    print("Saved:")
    print(loading_plot_path)


# ============================================================
# 16. 2D PCA PROJECTION
# ============================================================

if X_pca.shape[1] >= 2:

    plt.figure(figsize=(10, 7))

    if "PlacementStatus" in df.columns:

        placement = df["PlacementStatus"]

        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=placement,
            alpha=0.5
        )

        plt.colorbar(
            scatter,
            label="Placement Status"
        )

    else:

        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            alpha=0.5
        )

    plt.xlabel(
        f"PC1 ({explained_variance[0] * 100:.2f}%)"
    )

    plt.ylabel(
        f"PC2 ({explained_variance[1] * 100:.2f}%)"
    )

    plt.title("2D PCA Projection")

    plt.grid(True)

    plt.tight_layout()

    pca_2d_path = os.path.join(
        OUTPUT_DIR,
        "pca_2d_by_placement.png"
    )

    plt.savefig(
        pca_2d_path,
        dpi=300
    )

    plt.close()

    print("Saved:")
    print(pca_2d_path)

# ============================================================
# 17. PCA REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "pca_report.txt"
)

with open(report_path, "w") as f:

    f.write("PCA ANALYSIS REPORT\n")
    f.write("===================\n\n")

    f.write(
        f"Original dataset shape: {df.shape}\n"
    )

    f.write(
        f"Number of PCA features: {len(feature_columns)}\n\n"
    )

    f.write("Features used:\n")

    for feature in feature_columns:

        f.write(f"- {feature}\n")

    f.write("\nExplained Variance:\n")

    for i, value in enumerate(
        explained_variance
    ):

        f.write(
            f"PC{i + 1}: "
            f"{value * 100:.4f}%\n"
        )

    f.write("\nCumulative Variance:\n")

    for i, value in enumerate(
        cumulative_variance
    ):

        f.write(
            f"PC{i + 1}: "
            f"{value * 100:.4f}%\n"
        )

    f.write("\nComponent Requirements:\n")

    f.write(
        f"80% variance: {pc_80} components\n"
    )

    f.write(
        f"90% variance: {pc_90} components\n"
    )

    f.write(
        f"95% variance: {pc_95} components\n"
    )

