import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
# ---------------------------------------------------------
# 3. OUTPUT PATH
# ---------------------------------------------------------
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "Output",
    "plot"
)
os.makedirs(OUTPUT_PATH, exist_ok=True)
# ---------------------------------------------------------
# 4. LOAD DATASET
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# =========================================================
# PLOT 1: CGPA DISTRIBUTION
# =========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["CGPA"].dropna(),
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
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =========================================================
# PLOT 2: GENDER DISTRIBUTION
# =========================================================

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
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =========================================================
# PLOT 3: CGPA VS PLACEMENT STATUS
# =========================================================

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
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =========================================================
# PLOT 4: CGPA VS ATTENDANCE
# =========================================================

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
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =========================================================
# PLOT 5: CGPA AND INTERNSHIPS PAIRPLOT
# =========================================================

pair_data = df[
    ["CGPA", "Internships"]
].dropna()

pair_plot = sns.pairplot(
    pair_data,
    diag_kind="hist"
)

pair_plot.fig.suptitle(
    "CGPA and Internships Relationship",
    y=1.02
)

pair_plot.fig.savefig(
    os.path.join(
        OUTPUT_PATH,
        "cgpa_internships_pairplot.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close("all")


# =========================================================
# PLOT 6: CORRELATION HEATMAP
# =========================================================

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
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close()


# =========================================================
# DATASET INFORMATION
# =========================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nCGPA skewness:")
print(df["CGPA"].skew())

print("\nCGPA kurtosis:")
print(df["CGPA"].kurt())


# =========================================================
# COMPLETION MESSAGE
# =========================================================

print("\n========================================")
print("EDA COMPLETED SUCCESSFULLY")
print("========================================")

print("Dataset shape:", df.shape)

print("\nPlots saved in:")
print(OUTPUT_PATH)