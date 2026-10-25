from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

full_path = DATA_DIR / "gdsc2_with_smiles_full.csv"

df = pd.read_csv(full_path)

print("Full dataset rows:", len(df))
print("Unique drugs:", df["DRUG_ID"].nunique())
print("Unique cell lines:", df["CELL_LINE_NAME"].nunique())

# Split unique drugs, not individual rows
unique_drugs = df["DRUG_ID"].dropna().unique()

train_drugs, temp_drugs = train_test_split(
    unique_drugs,
    test_size=0.2,
    random_state=42
)

val_drugs, test_drugs = train_test_split(
    temp_drugs,
    test_size=0.5,
    random_state=42
)

train_df = df[df["DRUG_ID"].isin(train_drugs)].copy()
val_df = df[df["DRUG_ID"].isin(val_drugs)].copy()
test_df = df[df["DRUG_ID"].isin(test_drugs)].copy()

print("\nBefore sampling:")
print("Train rows:", len(train_df), "| drugs:", train_df["DRUG_ID"].nunique())
print("Val rows:", len(val_df), "| drugs:", val_df["DRUG_ID"].nunique())
print("Test rows:", len(test_df), "| drugs:", test_df["DRUG_ID"].nunique())

# CPU-friendly sample sizes
MAX_TRAIN = 24000
MAX_VAL = 3000
MAX_TEST = 3000

if len(train_df) > MAX_TRAIN:
    train_df = train_df.sample(n=MAX_TRAIN, random_state=42)

if len(val_df) > MAX_VAL:
    val_df = val_df.sample(n=MAX_VAL, random_state=42)

if len(test_df) > MAX_TEST:
    test_df = test_df.sample(n=MAX_TEST, random_state=42)

# Safety check: no drug overlap
train_set = set(train_df["DRUG_ID"])
val_set = set(val_df["DRUG_ID"])
test_set = set(test_df["DRUG_ID"])

print("\nDrug overlap check:")
print("Train-Val overlap:", len(train_set.intersection(val_set)))
print("Train-Test overlap:", len(train_set.intersection(test_set)))
print("Val-Test overlap:", len(val_set.intersection(test_set)))

train_df.to_csv(DATA_DIR / "cold_drug_train.csv", index=False)
val_df.to_csv(DATA_DIR / "cold_drug_val.csv", index=False)
test_df.to_csv(DATA_DIR / "cold_drug_test.csv", index=False)

print("\nAfter sampling:")
print("Train rows:", len(train_df), "| drugs:", train_df["DRUG_ID"].nunique())
print("Val rows:", len(val_df), "| drugs:", val_df["DRUG_ID"].nunique())
print("Test rows:", len(test_df), "| drugs:", test_df["DRUG_ID"].nunique())

print("\nSaved:")
print("data/cold_drug_train.csv")
print("data/cold_drug_val.csv")
print("data/cold_drug_test.csv")