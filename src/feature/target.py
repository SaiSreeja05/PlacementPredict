import os
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "placement_predict_50k Dataset (2).csv"
)


df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


target_column = "PlacementStatus"

target_features = [
    "Gender",
    "City",
    "Stream",
    "Specialisation"
]


# ============================================================
# 5. SELECT FIRST 10 ROWS
# ============================================================

data_10 = df[
    target_features + [target_column]
].head(10).copy()

print("\n==============================")
print("ORIGINAL FIRST 10 ROWS")
print("==============================")

print(data_10)


# ============================================================
# 6. TARGET ENCODING
# ============================================================

encoded_df = data_10.copy()

for column in target_features:

    # Calculate mean target value
    # for each category using the complete dataset
    mean_encoding = df.groupby(
        column
    )[target_column].mean()

    # Replace categories with
    # their corresponding mean target value
    encoded_df[column] = encoded_df[
        column
    ].map(mean_encoding)


# ============================================================
# 7. DISPLAY TARGET ENCODED DATA
# ============================================================

print("\n==============================")
print("TARGET ENCODED DATA")
print("==============================")

print(encoded_df)


# ============================================================
# 8. DISPLAY TARGET ENCODING VALUES
# ============================================================

print("\n==============================")
print("TARGET ENCODING VALUES")
print("==============================")

for column in target_features:

    mean_encoding = df.groupby(
        column
    )[target_column].mean()

    print("\n", column)
    print(mean_encoding)


# ============================================================
# 9. OUTPUT DIRECTORY
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
# 10. SAVE ORIGINAL DATA
# ============================================================

data_10.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "target_original_first_10.csv"
    ),
    index=False
)


# ============================================================
# 11. SAVE TARGET ENCODED DATA
# ============================================================

encoded_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "target_encoded_first_10.csv"
    ),
    index=False
)


# ============================================================
# 12. COMPLETION MESSAGE
# ============================================================

print("\n================================")
print("TARGET ENCODING COMPLETED")
print("================================")

print("Results saved in:")
print(OUTPUT_DIR)