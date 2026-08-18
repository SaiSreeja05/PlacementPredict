import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. PROJECT ROOT DIRECTORY
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
# 4. NOMINAL FEATURES
# ============================================================

nominal_features = [
    "Gender",
    "City",
    "Stream",
    "Specialisation"
]


# ============================================================
# 5. TAKE FIRST 10 ROWS
# ============================================================

data_10 = df[
    nominal_features
].head(10).copy()

print("\n==============================")
print("ORIGINAL FIRST 10 ROWS")
print("==============================")

print(data_10)


# ============================================================
# 6. ONE-HOT ENCODING
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded_data = encoder.fit_transform(
    data_10
)


# ============================================================
# 7. CREATE ENCODED DATAFRAME
# ============================================================

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(
        nominal_features
    )
)


print("\n==============================")
print("NOMINAL / ONE-HOT ENCODED DATA")
print("==============================")

print(encoded_df)


# ============================================================
# 8. OUTPUT DIRECTORY
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
# 9. SAVE ORIGINAL DATA
# ============================================================

data_10.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nominal_original_first_10.csv"
    ),
    index=False
)


# ============================================================
# 10. SAVE ENCODED DATA
# ============================================================

encoded_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nominal_encoded_first_10.csv"
    ),
    index=False
)


# ============================================================
# 11. COMPLETION MESSAGE
# ============================================================

print("\n================================")
print("NOMINAL ENCODING COMPLETED")
print("================================")

print("Results saved in:")
print(OUTPUT_DIR)