from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_DIR / "data" / "idukki_landslide_dataset.csv"
MODEL_DIR = PROJECT_DIR / "model"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. SENSOR FEATURES
# IMPORTANT: ORDER MUST MATCH THE ESP32 CODE
# Rainfall is intentionally excluded.
# ============================================================

FEATURES = [
    "soil_moisture_vwc_pct",
    "accel_vib_rms_g",
    "accel_peak_mag_g",
    "tilt_pitch_deg",
    "tilt_roll_deg",
]

TARGET = "hazard_label"

CLASS_NAMES = {
    0: "Lower Chance of Landslide",
    1: "Lookout for Landslide",
    2: "High Risk for Landslide",
}

CLASS_IDS = [0, 1, 2]

# ============================================================
# 3. LOAD AND VALIDATE DATASET
# ============================================================

print("=" * 65)
print("RANDOM FOREST LANDSLIDE CLASSIFICATION")
print("=" * 65)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

required_columns = FEATURES + [TARGET, "timestamp"]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="raise"
)

df = df.sort_values("timestamp").reset_index(drop=True)

# Convert model inputs and target to numeric values.
for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="raise")

if df[FEATURES + [TARGET]].isnull().any().any():
    raise ValueError(
        "Missing values detected in features or target."
    )

if not np.isfinite(
    df[FEATURES + [TARGET]].to_numpy(dtype=float)
).all():
    raise ValueError(
        "Non-finite values detected in features or target."
    )

if set(df[TARGET].unique()) != set(CLASS_IDS):
    raise ValueError(
        f"Expected labels {CLASS_IDS}, "
        f"found {sorted(df[TARGET].unique())}"
    )

X = df[FEATURES]
y = df[TARGET].astype(int)

print("\nDataset loaded successfully.")
print("Total records:", len(df))
print("Number of input features:", len(FEATURES))

print("\nFeatures used:")
for index, feature in enumerate(FEATURES):
    print(f"{index}: {feature}")

print("\nRainfall feature excluded: rainfall_24h")
print("\nOverall class distribution:")
print(y.value_counts().sort_index())

# ============================================================
# 4. CHRONOLOGICAL TRAIN / TEST SPLIT
# First 80% for training, final 20% for testing.
# ============================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print("\n" + "-" * 65)
print("DATA SPLIT")
print("-" * 65)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(
    "Training period:",
    df["timestamp"].iloc[0],
    "to",
    df["timestamp"].iloc[split_index - 1],
)
print(
    "Testing period:",
    df["timestamp"].iloc[split_index],
    "to",
    df["timestamp"].iloc[-1],
)

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())

if set(y_train.unique()) != set(CLASS_IDS):
    raise ValueError(
        "Training data does not contain all three classes. "
        "Review the chronological split before training."
    )

# ============================================================
# 5. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "-" * 65)
print("TRAINING RANDOM FOREST")
print("-" * 65)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Training completed.")
print("Number of trees:", model.n_estimators)
print("Maximum tree depth:", model.max_depth)

# ============================================================
# 6. EVALUATE THE MODEL
# ============================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(
    y_test, predictions
)

macro_precision = precision_score(
    y_test,
    predictions,
    labels=CLASS_IDS,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    y_test,
    predictions,
    labels=CLASS_IDS,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    y_test,
    predictions,
    labels=CLASS_IDS,
    average="macro",
    zero_division=0,
)

print("\n" + "=" * 65)
print("TEST SET PERFORMANCE")
print("=" * 65)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")
print(f"Macro precision:   {macro_precision:.4f}")
print(f"Macro recall:      {macro_recall:.4f}")
print(f"Macro F1-score:    {macro_f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=CLASS_IDS,
        target_names=[
            f"Class {i}: {CLASS_NAMES[i]}"
            for i in CLASS_IDS
        ],
        zero_division=0,
        digits=4,
    )
)

cm = confusion_matrix(
    y_test, predictions, labels=CLASS_IDS
)

print("Confusion matrix (rows = actual, columns = predicted):")
print(cm)

# ============================================================
# 7. GRAPH 1: OVERALL PERFORMANCE METRICS
# ============================================================

metric_names = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1-score",
]

metric_values = [
    accuracy,
    macro_precision,
    macro_recall,
    macro_f1,
]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(metric_names, metric_values)

ax.set_title("Random Forest - Overall Test Performance")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.35)

for bar, value in zip(bars, metric_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )

plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "overall_metrics.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 8. GRAPH 2: PRECISION, RECALL AND F1 FOR EACH CLASS
# ============================================================

report = classification_report(
    y_test,
    predictions,
    labels=CLASS_IDS,
    output_dict=True,
    zero_division=0,
)

class_precision = [
    report[str(i)]["precision"] for i in CLASS_IDS
]
class_recall = [
    report[str(i)]["recall"] for i in CLASS_IDS
]
class_f1 = [
    report[str(i)]["f1-score"] for i in CLASS_IDS
]

x = np.arange(len(CLASS_IDS))
width = 0.24

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    x - width,
    class_precision,
    width,
    label="Precision",
)
ax.bar(
    x,
    class_recall,
    width,
    label="Recall",
)
ax.bar(
    x + width,
    class_f1,
    width,
    label="F1-score",
)

ax.set_title("Performance by Hazard Class")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(
    [
        f"Class {i}\n{CLASS_NAMES[i]}"
        for i in CLASS_IDS
    ]
)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "class_metrics.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 9. GRAPH 3: FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_
order = np.argsort(importance)

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    np.array(FEATURES)[order],
    importance[order],
)

ax.set_title("Random Forest - Feature Importance")
ax.set_xlabel("Importance score")
ax.grid(axis="x", linestyle="--", alpha=0.35)

for bar, value in zip(bars, importance[order]):
    ax.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
    )

ax.set_xlim(0, max(importance) * 1.2 + 0.01)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print("\nFeature importance:")
for index in np.argsort(importance)[::-1]:
    print(f"{FEATURES[index]}: {importance[index]:.4f}")

# ============================================================
# 10. GRAPH 4: CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(image, ax=ax, label="Number of observations")

ax.set_title("Confusion Matrix - Test Set")
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")

ax.set_xticks(CLASS_IDS)
ax.set_yticks(CLASS_IDS)

ax.set_xticklabels(
    [f"Class {i}" for i in CLASS_IDS]
)
ax.set_yticklabels(
    [f"Class {i}" for i in CLASS_IDS]
)

threshold = cm.max() / 2 if cm.size else 0

for row in range(cm.shape[0]):
    for col in range(cm.shape[1]):
        ax.text(
            col,
            row,
            str(cm[row, col]),
            ha="center",
            va="center",
            color=(
                "white"
                if cm[row, col] > threshold
                else "black"
            ),
            fontsize=12,
        )

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 11. SAVE THE MODEL AND METADATA
# ============================================================

model_path = MODEL_DIR / "random_forest.joblib"

model_bundle = {
    "model": model,
    "features": FEATURES,
    "classes": model.classes_.tolist(),
    "class_names": CLASS_NAMES,
    "metrics": {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    },
}

joblib.dump(model_bundle, model_path)

# Save numerical results for later comparison.
metrics_df = pd.DataFrame(
    {
        "metric": metric_names,
        "score": metric_values,
    }
)
metrics_df.to_csv(
    RESULTS_DIR / "overall_metrics.csv",
    index=False,
)

print("\n" + "=" * 65)
print("PROJECT OUTPUTS SAVED")
print("=" * 65)

print("\nTrained model:")
print(model_path)

print("\nGraphs:")
for filename in [
    "overall_metrics.png",
    "class_metrics.png",
    "feature_importance.png",
    "confusion_matrix.png",
]:
    print(RESULTS_DIR / filename)

print("\nMetrics CSV:")
print(RESULTS_DIR / "overall_metrics.csv")

print("\nClass meanings:")
for class_id, class_name in CLASS_NAMES.items():
    print(f"{class_id}: {class_name}")

print("\nTraining and evaluation complete.")
