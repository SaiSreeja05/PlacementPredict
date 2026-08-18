from sklearn.preprocessing import OneHotEncoder
import os
import pandas as pd
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
# 4. FIRST 10 ROWS
# ============================================================

df_10 = df.head(10).copy()

print("\n================================")
print("FIRST 10 ROWS")
print("================================")

print(df_10)


# ============================================================
# 5. OUTPUT DIRECTORY
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
# 6. SAVE FIRST 10 ROWS
# ============================================================

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "first_10_rows.csv"
)

df_10.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 7. CREATE ONE-HOT ENCODER
# ============================================================

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)


# ============================================================
# 8. ENCODE CITY
# ============================================================

city_encoded = encoder.fit_transform(
    df_10[["City"]]
)


# ============================================================
# 9. CREATE ENCODED DATAFRAME
# ============================================================

city_encoded_df = pd.DataFrame(
    city_encoded,
    columns=encoder.get_feature_names_out(
        ["City"]
    ),
    index=df_10.index
)


print("\n================================")
print("ONE-HOT ENCODED CITY")
print("================================")

print(city_encoded_df)


# ============================================================
# 10. CREATE FINAL DATAFRAME
# ============================================================

df_10_encoded = pd.concat(
    [
        df_10.drop(
            columns=["City"]
        ),
        city_encoded_df
    ],
    axis=1
)


print("\n================================")
print("FINAL DATA AFTER ONE-HOT ENCODING")
print("================================")

print(df_10_encoded)


# ============================================================
# 11. SAVE ENCODED DATA
# ============================================================

ENCODED_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "one_hot_city_first_10.csv"
)

df_10_encoded.to_csv(
    ENCODED_OUTPUT_PATH,
    index=False
)

print("\n================================")
print("ONE-HOT ENCODING COMPLETED")
print("================================")
