from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Project folders
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

gdsc_path = DATA_DIR / "GDSC2-dataset.csv"
smiles_path = DATA_DIR / "drug_smiles_mapping.csv"

# Load files
gdsc = pd.read_csv(gdsc_path)
smiles = pd.read_csv(smiles_path)

print("GDSC2 shape:", gdsc.shape)
print("SMILES mapping shape:", smiles.shape)

# Keep useful columns
needed_cols = [
    "COSMIC_ID",
    "CELL_LINE_NAME",
    "TCGA_DESC",
    "DRUG_ID",
    "DRUG_NAME",
    "LN_IC50",
    "AUC",
    "Z_SCORE",
    "PUTATIVE_TARGET",
    "PATHWAY_NAME",
]

gdsc = gdsc[needed_cols].copy()

# Remove rows without target
gdsc = gdsc.dropna(subset=["LN_IC50"])

# Keep only rows where SMILES exists
smiles = smiles[["DRUG_ID", "SMILES"]].dropna()
smiles = smiles.drop_duplicates(subset=["DRUG_ID"])

# Merge drug response data with SMILES
df = gdsc.merge(smiles, on="DRUG_ID", how="inner")

# Remove missing required values
df = df.dropna(subset=["SMILES", "CELL_LINE_NAME", "LN_IC50"])

# Average duplicate drug-cell line pairs if they exist
df = (
    df.groupby(
        ["DRUG_ID", "DRUG_NAME", "SMILES", "COSMIC_ID", "CELL_LINE_NAME", "TCGA_DESC"],
        as_index=False
    )
    .agg(
        LN_IC50=("LN_IC50", "mean"),
        AUC=("AUC", "mean"),
        Z_SCORE=("Z_SCORE", "mean"),
    )
)

print("\nAfter merging with SMILES:")
print("Rows:", len(df))
print("Unique drugs:", df["DRUG_ID"].nunique())
print("Unique cell lines:", df["CELL_LINE_NAME"].nunique())

# Start with smaller sample because you are training on CPU
MAX_ROWS = 30000

if len(df) > MAX_ROWS:
    df_sample = df.sample(n=MAX_ROWS, random_state=42)
else:
    df_sample = df.copy()

# Split data
train_df, temp_df = train_test_split(df_sample, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save outputs
df.to_csv(DATA_DIR / "gdsc2_with_smiles_full.csv", index=False)
df_sample.to_csv(DATA_DIR / "gdsc2_with_smiles_sample.csv", index=False)
train_df.to_csv(DATA_DIR / "train.csv", index=False)
val_df.to_csv(DATA_DIR / "val.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)

print("\nSaved files:")
print("data/gdsc2_with_smiles_full.csv")
print("data/gdsc2_with_smiles_sample.csv")
print("data/train.csv")
print("data/val.csv")
print("data/test.csv")

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))