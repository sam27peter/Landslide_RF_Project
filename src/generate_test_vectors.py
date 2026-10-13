from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_DIR / "data" / "idukki_landslide_dataset.csv"
MODEL_PATH = PROJECT_DIR / "model" / "random_forest.joblib"
OUTPUT_PATH = PROJECT_DIR / "results" / "cpp_test_vectors.csv"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

df = pd.read_csv(DATASET_PATH)

# Use representative records from all three actual classes.
samples = (
    df.groupby("hazard_label", group_keys=False)
    .sample(n=5, random_state=42)
    .sort_index()
)

X = samples[features]
predictions = model.predict(X)
probabilities = model.predict_proba(X)

result = X.copy()
result["actual_label"] = samples["hazard_label"].to_numpy()
result["predicted_class"] = predictions

for index, class_id in enumerate(model.classes_):
    result[f"p{class_id}"] = probabilities[:, index]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False)

print("Reference test vectors generated.")
print("Number of test cases:", len(result))
print("Output:", OUTPUT_PATH)
print("\nExpected predictions:")
print(result.to_string(index=False))
