import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "data" / "idukki_landslide_dataset.csv"

df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

features = [
    "rainfall_24h",
    "soil_moisture_vwc_pct",
    "accel_vib_rms_g",
    "accel_peak_mag_g",
    "tilt_pitch_deg",
    "tilt_roll_deg",
]

print("=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)

print("\nTime range:")
print("Start:", df["timestamp"].min())
print("End:  ", df["timestamp"].max())

print("\nEvent sources:")
print(df["event_source"].value_counts())

print("\nClass counts by month:")
df["month"] = df["timestamp"].dt.to_period("M")
print(pd.crosstab(df["month"], df["hazard_label"]))

print("\nFeature averages by hazard label:")
print(df.groupby("hazard_label")[features].mean().round(4))

print("\nFeature ranges:")
print(df[features].agg(["min", "max"]).T)

print("\nAnalysis complete.")