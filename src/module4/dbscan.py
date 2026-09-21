import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = r"C:\Users\psais\PycharmProjects\PlacementPrediction"
DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "placement_predict_50k Dataset (2).csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "module4", "outputs", "dbscan")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBSAMPLE_SIZE = 5000
MIN_SAMPLES = 40
RANDOM_STATE = 42


print("\n====================================")
print("        DBSCAN CLUSTERING")
print("====================================\n")

df_full = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df_full.shape}")


# ============================================================
# REMOVE NON-NUMERIC / TARGET COLUMNS
# ============================================================

DROP_COLUMNS = ["StudentID", "PlacementStatus", "Salary Package"]

drop_existing = [col for col in DROP_COLUMNS if col in df_full.columns]

df_features = df_full.drop(columns=drop_existing)

numeric_columns = df_features.select_dtypes(include=np.number).columns.tolist()

if "IsAnomaly" in numeric_columns:
    numeric_columns.remove("IsAnomaly")


print(f"\nNumeric features used for DBSCAN: {len(numeric_columns)}")

print("\nFeatures:")

for col in numeric_columns:
    print(" -", col)

x_raw_full = df_full[numeric_columns].copy()


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\nApplying median imputation...")

imputer = SimpleImputer(strategy="median")

x_imputed_full = imputer.fit_transform(x_raw_full)


# ============================================================
# STANDARDIZATION
# ============================================================

print("Applying StandardScaler...")

scaler = StandardScaler()

x_scaled_full = scaler.fit_transform(x_imputed_full)


# ============================================================
# SUBSAMPLE
# ============================================================

rng = np.random.RandomState(RANDOM_STATE)

sample_size = min(SUBSAMPLE_SIZE, len(x_scaled_full))

sample_idx = rng.choice(len(x_scaled_full), size=sample_size, replace=False)

x_scaled = x_scaled_full[sample_idx]

df = df_full.iloc[sample_idx].reset_index(drop=True)


print(f"\nFull dataset: {len(df_full):,} rows")
print(f"DBSCAN sample: {len(df):,} rows")


# ============================================================
# ISANOMALY INFORMATION
# ============================================================

if "IsAnomaly" in df.columns:
    anomaly_count = int(df["IsAnomaly"].sum())
    anomaly_percent = round(df["IsAnomaly"].mean() * 100, 2)

    print(f"\nIsAnomaly=1 in sample: {anomaly_count:,} ({anomaly_percent}%)")


# ============================================================
# K-DISTANCE PLOT
# ============================================================

print("\n====================================")
print("        K-DISTANCE ANALYSIS")
print("====================================\n")

neighbors = NearestNeighbors(n_neighbors=MIN_SAMPLES)

neighbors.fit(x_scaled)

distances, _ = neighbors.kneighbors(x_scaled)

k_distances = np.sort(distances[:, -1])


plt.figure(figsize=(9, 5))

plt.plot(k_distances)

plt.xlabel("Points sorted by k-distance")
plt.ylabel(f"Distance to {MIN_SAMPLES}th nearest neighbor")
plt.title(f"K-Distance Plot (min_samples={MIN_SAMPLES})")

plt.grid(alpha=0.3)

plt.tight_layout()

k_distance_path = os.path.join(OUTPUT_DIR, "k_distance_plot.png")

plt.savefig(k_distance_path, dpi=150)

plt.close()

print("K-distance plot saved:")
print(k_distance_path)


# ============================================================
# AUTOMATIC EPS
# ============================================================

n_points = len(k_distances)

x_norm = np.linspace(0, 1, n_points)

distance_range = k_distances.max() - k_distances.min()

if distance_range == 0:
    EPS = 0.5
else:
    y_norm = (k_distances - k_distances.min()) / distance_range

    numerator = np.abs((y_norm[-1] - y_norm[0]) * x_norm - (x_norm[-1] - x_norm[0]) * y_norm + x_norm[-1] * y_norm[0] - y_norm[-1] * x_norm[0])

    denominator = np.sqrt((y_norm[-1] - y_norm[0]) ** 2 + (x_norm[-1] - x_norm[0]) ** 2)

    if denominator == 0:
        EPS = 0.5
    else:
        perpendicular_distance = numerator / denominator

        elbow_idx = int(np.argmax(perpendicular_distance))

        EPS = round(float(k_distances[elbow_idx]), 2)


print(f"\nAutomatically selected EPS: {EPS}")


# ============================================================
# RUN DBSCAN
# ============================================================

print("\n====================================")
print("          RUNNING DBSCAN")
print("====================================\n")

dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, algorithm="ball_tree", n_jobs=1)

cluster_labels = dbscan.fit_predict(x_scaled)


# ============================================================
# RESULTS
# ============================================================

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

n_noise = int((cluster_labels == -1).sum())

n_core = len(dbscan.core_sample_indices_)


print("DBSCAN RESULTS")

print(f"Clusters found: {n_clusters}")
print(f"Core points: {n_core:,}")
print(f"Noise points: {n_noise:,}")
print(f"Noise percentage: {round(n_noise / len(df) * 100, 2)}%")


# ============================================================
# CLUSTER SIZES
# ============================================================

cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()

print("\nCluster sizes:")
print(cluster_sizes.to_string())


# ============================================================
# EPS SENSITIVITY
# ============================================================

print("\n====================================")
print("          EPS SENSITIVITY")
print("====================================\n")

eps_values = [round(EPS * 0.5, 2), EPS, round(EPS * 2, 2)]

sensitivity_results = []

for eps_value in eps_values:
    model = DBSCAN(eps=eps_value, min_samples=MIN_SAMPLES, algorithm="ball_tree", n_jobs=1)

    labels = model.fit_predict(x_scaled)

    clusters = len(set(labels)) - (1 if -1 in labels else 0)

    noise = int((labels == -1).sum())

    sensitivity_results.append({"eps": eps_value, "clusters": clusters, "noise": noise})

    print(f"eps={eps_value}: {clusters} clusters, {noise:,} noise points")


# ============================================================
# PCA VISUALIZATION
# ============================================================

print("\n====================================")
print("        PCA VISUALIZATION")
print("====================================\n")

pca = PCA(n_components=2, random_state=RANDOM_STATE)

x_2d = pca.fit_transform(x_scaled)


plt.figure(figsize=(9, 7))

noise_mask = cluster_labels == -1

cluster_mask = cluster_labels != -1


if cluster_mask.any():
    plt.scatter(x_2d[cluster_mask, 0], x_2d[cluster_mask, 1], c=cluster_labels[cluster_mask], cmap="tab10", s=8, alpha=0.6, label="Clusters")


if noise_mask.any():
    plt.scatter(x_2d[noise_mask, 0], x_2d[noise_mask, 1], c="black", marker="x", s=20, alpha=0.7, label="Noise")


plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title(f"DBSCAN Clusters (eps={EPS}, min_samples={MIN_SAMPLES})")

plt.legend()

plt.tight_layout()

pca_path = os.path.join(OUTPUT_DIR, "dbscan_clusters_pca_2d.png")

plt.savefig(pca_path, dpi=150)

plt.close()

print("PCA plot saved:")
print(pca_path)


# ============================================================
# ANOMALY COMPARISON
# ============================================================

both_flagged = 0
noise_that_is_anomaly = 0
anomaly_that_is_noise = 0

if "IsAnomaly" in df.columns:
    is_noise = cluster_labels == -1

    anomaly_flag = df["IsAnomaly"].values.astype(bool)

    both_flagged = int((is_noise & anomaly_flag).sum())

    noise_that_is_anomaly = round(both_flagged / max(n_noise, 1) * 100, 1)

    anomaly_that_is_noise = round(both_flagged / max(anomaly_flag.sum(), 1) * 100, 1)

    print("\n====================================")
    print("       ANOMALY COMPARISON")
    print("====================================")

    print(f"Both noise and anomalous: {both_flagged:,}")
    print(f"DBSCAN noise that is anomalous: {noise_that_is_anomaly}%")
    print(f"Anomalies detected as noise: {anomaly_that_is_noise}%")


# ============================================================
# PLACEMENT RATE
# ============================================================

placement_clustered = np.nan
placement_noise = np.nan

if "PlacementStatus" in df.columns:
    placement_clustered = df.loc[cluster_mask, "PlacementStatus"].mean() if cluster_mask.any() else np.nan

    placement_noise = df.loc[noise_mask, "PlacementStatus"].mean() if noise_mask.any() else np.nan

    print("\n====================================")
    print("        PLACEMENT RATE")
    print("====================================")

    if not np.isnan(placement_clustered):
        print(f"Clustered students: {round(placement_clustered * 100, 1)}%")

    if not np.isnan(placement_noise):
        print(f"Noise students: {round(placement_noise * 100, 1)}%")


# ============================================================
# SAVE REPORT
# ============================================================

report_path = os.path.join(OUTPUT_DIR, "dbscan_report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("DBSCAN CLUSTERING REPORT\n")
    f.write("========================\n\n")

    f.write(f"Dataset: {DATA_PATH}\n")
    f.write(f"Full dataset rows: {len(df_full):,}\n")
    f.write(f"DBSCAN sample rows: {len(df):,}\n")
    f.write(f"Number of features: {len(numeric_columns)}\n\n")

    f.write(f"min_samples: {MIN_SAMPLES}\n")
    f.write(f"eps: {EPS}\n\n")

    f.write(f"Clusters found: {n_clusters}\n")
    f.write(f"Core points: {n_core:,}\n")
    f.write(f"Noise points: {n_noise:,}\n")
    f.write(f"Noise percentage: {round(n_noise / len(df) * 100, 2)}%\n\n")

    f.write("Cluster sizes:\n")
    f.write(cluster_sizes.to_string())
    f.write("\n\n")

    if "IsAnomaly" in df.columns:
        f.write("ANOMALY COMPARISON\n")
        f.write(f"Both noise and anomalous: {both_flagged:,}\n")
        f.write(f"DBSCAN noise that is anomalous: {noise_that_is_anomaly}%\n")
        f.write(f"Anomalies detected as noise: {anomaly_that_is_noise}%\n\n")

    f.write("EPS SENSITIVITY\n")

    for result in sensitivity_results:
        f.write(f"eps={result['eps']}: {result['clusters']} clusters, {result['noise']:,} noise points\n")

    f.write("\n")

    if not np.isnan(placement_clustered):
        f.write(f"Placement rate - clustered: {round(placement_clustered * 100, 1)}%\n")

    if not np.isnan(placement_noise):
        f.write(f"Placement rate - noise: {round(placement_noise * 100, 1)}%\n")


print("\nReport saved:")
print(report_path)

print("\n====================================")
print("       DBSCAN COMPLETED")
print("====================================")

print("\nFiles generated:")
print("1. k_distance_plot.png")
print("2. dbscan_clusters_pca_2d.png")
print("3. dbscan_report.txt")
print()