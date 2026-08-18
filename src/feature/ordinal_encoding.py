import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


# --------------------------------------------------
# 1. PROJECT DIRECTORY
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# --------------------------------------------------
# 2. DATASET PATH
# --------------------------------------------------

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "placement_predict_50k Dataset (2).csv"
)


# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 4. SELECT FIRST 10 ROWS
# --------------------------------------------------

df_10 = df.head(10).copy()

print("\n================================")
print("FIRST 10 ROWS")
print("================================")

print(df_10)


# --------------------------------------------------
# 5. SELECT ORDINAL COLUMN
# --------------------------------------------------

ordinal_data = df_10[
    ["CGPA_Tier"]
].copy()

print("\n================================")
print("ORIGINAL CGPA_Tier")
print("================================")

print(ordinal_data)


# --------------------------------------------------
# 6. ORDINAL ENCODING
# --------------------------------------------------

encoder = OrdinalEncoder(
    categories=[
        ["Low", "Mid", "High"]
    ]
)

ordinal_encoded = encoder.fit_transform(
    ordinal_data
)


# --------------------------------------------------
# 7. CONVERT TO DATAFRAME
# --------------------------------------------------

ordinal_encoded = pd.DataFrame(
    ordinal_encoded,
    columns=["CGPA_Tier"],
    index=df_10.index
)


# --------------------------------------------------
# 8. DISPLAY RESULT
# --------------------------------------------------

print("\n================================")
print("ORDINAL ENCODED DATA")
print("================================")

print(ordinal_encoded)


# --------------------------------------------------
# 9. OUTPUT DIRECTORY
# --------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Output",
    "feature_engineering"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# 10. SAVE RESULT
# --------------------------------------------------

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "ordinal_encoded_first_10.csv"
)

ordinal_encoded.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 11. COMPLETION MESSAGE
# --------------------------------------------------

print("\n================================")
print("ORDINAL ENCODING COMPLETED")
print("================================")

print("Results saved in:")
print(OUTPUT_PATH)