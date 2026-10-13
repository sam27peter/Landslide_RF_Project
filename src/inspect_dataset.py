import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "idukki_landslide_dataset.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("IDUKKI LANDSLIDE DATASET INSPECTION")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nColumn names:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

if "hazard_label" in df.columns:
    print("\nHazard label counts:")
    print(df["hazard_label"].value_counts(dropna=False).sort_index())

    print("\nHazard label percentages:")
    print(
        (df["hazard_label"].value_counts(normalize=True, dropna=False)
         * 100).round(2)
    )
else:
    print("\nWARNING: 'hazard_label' column not found.")

print("\nDataset inspection complete.")