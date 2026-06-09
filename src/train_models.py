"""Train and save ML models. Called automatically if model files are missing."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.capacity_calculator import calculate_wallbox_capacity

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"

ZONE_FEATURES = [
    "residential_index_derived",
    "grid_sensitivity_index_derived",
    "reserve_capacity_kw_2025_synthetic",
    "reserve_margin_pct_2025_synthetic",
    "no_private_parking_index_derived",
]

CAPACITY_FEATURES = [
    "available_for_ev_kw",
    "max_wallboxes_without_balancing",
    "max_simultaneous_with_balancing",
    "total_building_capacity_kw",
]

BUILDING_FEATURES = [
    "expected_evs",
    "parking_spaces",
    "flats",
]

ALL_FEATURES = ZONE_FEATURES + CAPACITY_FEATURES + BUILDING_FEATURES

BREAKER_CHOICES = [25, 32, 40, 50, 63, 80, 100, 125, 160]
SAMPLES_PER_ZONE = 20


def _derive_labels(available, reserve_margin, grid_sens, expected_evs):
    """Deterministically derive solution_label and lb_label from capacity + zone params."""
    if available < 7:
        label = "none_monitor"
        lb_label = "none"
    elif available < 22 or expected_evs <= 2:
        label = "residential_ac_small"
        lb_label = "static" if grid_sens < 0.6 else "dynamic"
    elif available < 55 or expected_evs <= 8:
        label = "residential_ac_medium"
        lb_label = "dynamic" if grid_sens > 0.5 else "static"
    else:
        label = "residential_ac_medium"  # Cap at medium for residential buildings
        lb_label = "dynamic"

    # Override: if reserve_margin < 10, downgrade one level
    if reserve_margin < 10 and label == "residential_ac_medium":
        label = "residential_ac_small"

    return label, lb_label


def generate_synthetic_training_data(zones_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []

    for _, zone in zones_df.iterrows():
        reserve_margin = float(zone.get("reserve_margin_pct_2025_synthetic", 20))
        grid_sens = float(zone.get("grid_sensitivity_index_derived", 0.5))

        for _ in range(SAMPLES_PER_ZONE):
            flats = int(rng.integers(4, 151))
            parking_spaces = int(rng.integers(0, flats + 1))
            breaker_amps = int(rng.choice(BREAKER_CHOICES))
            max_evs = min(parking_spaces + 1, 50)
            expected_evs = int(rng.integers(0, max_evs + 1))
            has_pv = bool(rng.random() < 0.3)
            pv_kwp = float(rng.uniform(5, 50)) if has_pv else 0.0

            cap = calculate_wallbox_capacity(
                flats, parking_spaces, breaker_amps, expected_evs, has_pv, pv_kwp
            )

            available = cap["available_for_ev_kw"]
            label, lb_label = _derive_labels(available, reserve_margin, grid_sens, expected_evs)

            row = {feat: zone[feat] for feat in ZONE_FEATURES}
            row["available_for_ev_kw"] = available
            row["max_wallboxes_without_balancing"] = cap["max_wallboxes_without_balancing"]
            row["max_simultaneous_with_balancing"] = cap["max_simultaneous_with_balancing"]
            row["total_building_capacity_kw"] = cap["total_building_capacity_kw"]
            row["expected_evs"] = expected_evs
            row["parking_spaces"] = parking_spaces
            row["flats"] = flats
            row["solution_label"] = label
            row["lb_label"] = lb_label
            rows.append(row)

    return pd.DataFrame(rows)


def train_classifier():
    train_df = pd.read_csv(DATA_DIR / "zones_train.csv")
    val_df = pd.read_csv(DATA_DIR / "zones_validation.csv")
    zones_df = pd.concat([train_df, val_df], ignore_index=True)

    synthetic_df = generate_synthetic_training_data(zones_df)
    print(f"Generated {len(synthetic_df)} synthetic training samples")
    print(f"Solution label distribution:\n{synthetic_df['solution_label'].value_counts()}")
    print(f"LB label distribution:\n{synthetic_df['lb_label'].value_counts()}")

    X = synthetic_df[ALL_FEATURES].fillna(0)
    y_sol = synthetic_df["solution_label"]
    y_lb = synthetic_df["lb_label"]

    solution_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    solution_clf.fit(X, y_sol)

    lb_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    lb_clf.fit(X, y_lb)

    # Quick self-accuracy check
    sol_acc = accuracy_score(y_sol, solution_clf.predict(X))
    lb_acc = accuracy_score(y_lb, lb_clf.predict(X))
    print(f"Solution classifier train accuracy: {sol_acc:.3f}")
    print(f"LB classifier train accuracy: {lb_acc:.3f}")

    MODELS_DIR.mkdir(exist_ok=True)

    # Remove old classifier if it exists
    old_path = MODELS_DIR / "classifier.joblib"
    if old_path.exists():
        old_path.unlink()
        print("Deleted old models/classifier.joblib")

    joblib.dump(solution_clf, MODELS_DIR / "solution_classifier.joblib")
    joblib.dump(lb_clf, MODELS_DIR / "lb_classifier.joblib")
    print("Saved models/solution_classifier.joblib and models/lb_classifier.joblib")

    return solution_clf, lb_clf


def load_or_train_classifier():
    sol_path = MODELS_DIR / "solution_classifier.joblib"
    lb_path = MODELS_DIR / "lb_classifier.joblib"
    if sol_path.exists() and lb_path.exists():
        return joblib.load(sol_path), joblib.load(lb_path)
    print("Models not found — training classifiers…")
    return train_classifier()


def predict_solution(
    model_tuple,
    zone_row: pd.Series,
    capacity_result: dict,
    user_input: dict,
) -> dict:
    solution_clf, lb_clf = model_tuple

    feature_values = {feat: zone_row.get(feat, 0) for feat in ZONE_FEATURES}
    feature_values["available_for_ev_kw"] = capacity_result["available_for_ev_kw"]
    feature_values["max_wallboxes_without_balancing"] = capacity_result["max_wallboxes_without_balancing"]
    feature_values["max_simultaneous_with_balancing"] = capacity_result["max_simultaneous_with_balancing"]
    feature_values["total_building_capacity_kw"] = capacity_result["total_building_capacity_kw"]
    feature_values["expected_evs"] = user_input.get("expected_evs", 0)
    feature_values["parking_spaces"] = user_input.get("parking_spaces", 0)
    feature_values["flats"] = user_input.get("flats", 0)

    X = pd.DataFrame([feature_values], columns=ALL_FEATURES).fillna(0)

    solution_type = solution_clf.predict(X)[0]
    load_balancing_type = lb_clf.predict(X)[0]

    return {
        "solution_type": solution_type,
        "load_balancing_type": load_balancing_type,
    }


if __name__ == "__main__":
    train_classifier()
    print("Done.")
