import os

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "placement_predict_50k Dataset (2).csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "Output",
    "plot"
)

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# 1. CGPA Distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["CGPA"],
    bins=10,
    edgecolor="black"
)

plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "cgpa_distribution.png"
    ),
    dpi=150
)

plt.close()


# --------------------------------------------------
# 2. Gender Distribution
# --------------------------------------------------

plt.figure(figsize=(7, 5))

df["Gender"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Gender Distribution")
plt.ylabel("")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "gender_distribution.png"
    ),
    dpi=150
)

plt.close()


# --------------------------------------------------
# 3. CGPA vs Placement Status
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="PlacementStatus",
    y="CGPA",
    data=df
)

plt.title("CGPA vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("CGPA")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "cgpa_vs_placement.png"
    ),
    dpi=150
)

plt.close()


# --------------------------------------------------
# 4. CGPA vs Attendance
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x="CGPA",
    y="AttendancePercent",
    data=df
)

plt.title("CGPA vs Attendance")
plt.xlabel("CGPA")
plt.ylabel("Attendance Percent")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "cgpa_vs_attendance.png"
    ),
    dpi=150
)

plt.close()


# --------------------------------------------------
# 5. CGPA and Internships Pairplot
# --------------------------------------------------

pair_data = df[
    ["CGPA", "Internships"]
].dropna()

sns.pairplot(
    pair_data,
    diag_kind="hist"
)

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "cgpa_internships_pairplot.png"
    ),
    dpi=150
)

plt.close("all")


# --------------------------------------------------
# 6. Correlation Heatmap
# --------------------------------------------------

numeric_df = df.select_dtypes(
    include=["number"]
)

corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "correlation_heatmap.png"
    ),
    dpi=150
)

plt.close()


# --------------------------------------------------
# EDA information
# --------------------------------------------------

print("EDA completed successfully.")

print("Dataset shape:", df.shape)

print("CGPA skewness:", df["CGPA"].skew())

print("CGPA kurtosis:", df["CGPA"].kurt())

print("Plots saved in:")
print(OUTPUT_PATH)