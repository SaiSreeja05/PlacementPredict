import os
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# ============================================================
# 2. DATASET PATH
# ============================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "placement_predict_50k Dataset (2).csv"
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 4. FEATURES
# ============================================================

features = [
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore"
]


# ============================================================
# 5. SELECT FIRST 10 ROWS
# ============================================================

data_10 = df[
    features
].head(10).copy()

print("\n==============================")
print("FIRST 10 ROWS")
print("==============================")

print(data_10)


# ============================================================
# 6. STANDARD SCALING
# ============================================================

standard_scaler = StandardScaler()

standard_scaled = (
    standard_scaler.fit_transform(
        data_10
    )
)

standard_df = pd.DataFrame(
    standard_scaled,
    columns=features
)

print("\n==============================")
print("STANDARDIZED DATA")
print("==============================")

print(standard_df)


# ============================================================
# 7. MIN-MAX SCALING
# ============================================================

minmax_scaler = MinMaxScaler()

minmax_scaled = (
    minmax_scaler.fit_transform(
        data_10
    )
)

minmax_df = pd.DataFrame(
    minmax_scaled,
    columns=features
)

print("\n==============================")
print("MIN-MAX SCALED DATA")
print("==============================")

print(minmax_df)


# ============================================================
# 8. ROBUST SCALING
# ============================================================

robust_scaler = RobustScaler()

robust_scaled = (
    robust_scaler.fit_transform(
        data_10
    )
)

robust_df = pd.DataFrame(
    robust_scaled,
    columns=features
)

print("\n==============================")
print("ROBUST SCALED DATA")
print("==============================")

print(robust_df)


# ============================================================
# 9. STANDARDIZED MEAN
# ============================================================

print("\n==============================")
print("STANDARDIZED MEAN")
print("==============================")

print(
    standard_df.mean()
)


# ============================================================
# 10. STANDARDIZED STD
# ============================================================

print("\n==============================")
print("STANDARDIZED STD")
print("==============================")

print(
    standard_df.std()
)


# ============================================================
# 11. STANDARD SCALER MEAN
# ============================================================

print("\n==============================")
print("STANDARD SCALER MEAN")
print("==============================")

print(
    standard_scaler.mean_
)


# ============================================================
# 12. STANDARD SCALER SCALE
# ============================================================

print("\n==============================")
print("STANDARD SCALER SCALE")
print("==============================")

print(
    standard_scaler.scale_
)


# ============================================================
# 13. OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Output",
    "feature_engineering"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 14. SAVE ORIGINAL DATA
# ============================================================

data_10.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "original_first_10.csv"
    ),
    index=False
)


# ============================================================
# 15. SAVE STANDARDIZED DATA
# ============================================================

standard_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "standard_scaled_first_10.csv"
    ),
    index=False
)


# ============================================================
# 16. SAVE MIN-MAX DATA
# ============================================================

minmax_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "minmax_scaled_first_10.csv"
    ),
    index=False
)


# ============================================================
# 17. SAVE ROBUST DATA
# ============================================================

robust_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "robust_scaled_first_10.csv"
    ),
    index=False
)


# ============================================================
# 18. COMPLETION MESSAGE
# ============================================================

print("\n================================")
print("FEATURE ENGINEERING COMPLETED")
print("================================")

print(
    "Results saved in:"
)

print(
    OUTPUT_DIR
)