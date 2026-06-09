"""Train and save ML models. Called automatically if model files are missing."""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"

CLASSIFIER_FEATURES = [
    "residential_index_derived",
    "destination_activity_index_derived",
    "no_private_parking_index_derived",
    "grid_sensitivity_index_derived",
    "reserve_capacity_kw_2025_synthetic",
    "reserve_margin_pct_2025_synthetic",
    "target_daily_charging_kwh_2030_synthetic",
    "target_peak_charging_kw_2030_synthetic",
    "charging_points_2026_real",
    "charging_station_power_kw_2026_real",
    "parking_segments_zps_real",
    "population_model_input_derived",
]

CLASSIFIER_TARGET = "target_recommended_solution_synthetic"


def train_classifier():
    train_df = pd.read_csv(DATA_DIR / "zones_train.csv")
    val_df = pd.read_csv(DATA_DIR / "zones_validation.csv")

    X_train = train_df[CLASSIFIER_FEATURES].fillna(0)
    y_train = train_df[CLASSIFIER_TARGET]
    X_val = val_df[CLASSIFIER_FEATURES].fillna(0)
    y_val = val_df[CLASSIFIER_TARGET]

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Classifier validation accuracy: {acc:.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "classifier.joblib")
    return model


def load_or_train_classifier():
    path = MODELS_DIR / "classifier.joblib"
    if path.exists():
        return joblib.load(path)
    print("Model not found — training classifier…")
    return train_classifier()


def predict_solution(model, zone_row: pd.Series) -> str:
    features = pd.DataFrame(
        [zone_row[CLASSIFIER_FEATURES].infer_objects().fillna(0).values],
        columns=CLASSIFIER_FEATURES,
    )
    return model.predict(features)[0]


if __name__ == "__main__":
    train_classifier()
    print("Done.")
