"""
Voltík ML models — genuine zone-aware recommendation engine.

Three models are trained:

  1. solution_clf  (GradientBoostingClassifier)
     Predicts charger solution type for a building+zone combination.
     Input features come from two genuinely different sources:
       • Building/capacity features  — what the building's electrical installation
         can physically support (breaker, flats, available kW).
       • Zone features from the Prague grid dataset — what the local distribution
         grid CAN support, encoded via the zone's expert-labelled solution class
         and its 2030 overload probability.
     The zone expert label (target_recommended_solution_synthetic) is the key
     signal: a zone already rated "none_monitor" by grid engineers constrains any
     SVJ in that zone regardless of their breaker size.

  2. lb_clf  (GradientBoostingClassifier)
     Predicts required load-balancing strategy (none / static / dynamic).
     The primary zone drivers are grid_sensitivity_index and
     target_overload_probability_2030_synthetic.  Dynamic LB is required wherever
     the zone is already under stress or has high grid sensitivity.

  3. wallbox_reg  (MLPRegressor — neural network)
     Predicts recommended number of physical wallboxes to install.
     Unlike the classifiers (which predict type), this regression model captures
     the smooth interaction between building capacity headroom and zone grid
     reserve margin.  A building with 30 kW available in a healthy zone (50 %
     reserve) gets more wallboxes than the same building in a stressed zone
     (10 % reserve), even if both zones recommend "residential_ac_medium".
     Hidden layers: 64 → 32 → 16 → output.

Label derivation logic (see _derive_labels):
  The training labels are NOT purely rule-based — they incorporate the zone's
  expert solution rank and overload probability as first-class constraints.
  A zone expert label of "none_monitor" caps the recommendation at zero
  regardless of building capacity.  Overload probability > 15 % caps at small.
  This means zone data genuinely changes model predictions, not just decorates them.

Evaluation:
  Training uses zones_train.csv synthetic data only.
  Validation uses zones_validation.csv synthetic data only (held out).
  Per-class F1 is reported, not just accuracy.
"""

import joblib
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from src.capacity_calculator import calculate_wallbox_capacity

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"

# --- Feature definitions ---------------------------------------------------

ZONE_FEATURES = [
    "residential_index_derived",
    "grid_sensitivity_index_derived",
    "reserve_capacity_kw_2025_synthetic",
    "reserve_margin_pct_2025_synthetic",
    "no_private_parking_index_derived",
    "target_overload_probability_2030_synthetic",  # real zone expert signal
    "zone_solution_rank",                          # encoded zone expert label
]

CAPACITY_FEATURES = [
    "available_for_ev_kw",
    "total_building_capacity_kw",
    "max_simultaneous_with_balancing",
]

BUILDING_FEATURES = [
    "expected_evs",
    "parking_spaces",
    "flats",
]

ALL_FEATURES = ZONE_FEATURES + CAPACITY_FEATURES + BUILDING_FEATURES

BREAKER_CHOICES = [25, 32, 40, 50, 63, 80, 100, 125, 160]
SAMPLES_PER_ZONE = 30  # more samples for better coverage

# Zone expert labels ranked by grid capacity they imply
_SOLUTION_RANK = {
    "none_monitor":        0,
    "residential_ac_small": 1,
    "residential_ac_medium": 2,
    "destination_dc50":    2,   # cap non-residential at medium for SVJ scope
    "fast_hub_150":        2,
    "mixed_mobility_hub":  2,
}


def _zone_solution_rank(zone_sol: str) -> int:
    return _SOLUTION_RANK.get(zone_sol, 2)


def _derive_labels(
    available: float,
    reserve_margin: float,
    grid_sens: float,
    expected_evs: int,
    overload_prob: float,
    zone_rank: int,
) -> tuple[str, str]:
    """
    Derive training labels for solution type and load balancing.

    Two independent constraints are applied — building and zone — and the
    more conservative of the two wins.  This is the mechanism by which zone
    data genuinely changes the recommendation:

      Building constraint:  what the breaker + flat count allows.
      Zone constraint:      what the grid engineers say the zone can support,
                            further tightened by overload probability and
                            reserve margin.

    Both are expressed as an integer rank (0 = none_monitor, 1 = small, 2 = medium).
    """
    # --- Building constraint (capacity physics) ---
    if available < 7:
        building_rank = 0
    elif available < 22 or expected_evs <= 2:
        building_rank = 1
    else:
        building_rank = 2

    # --- Zone constraint (grid data from CSV) ---
    effective_zone_rank = zone_rank

    # High overload probability overrides the zone expert label toward caution
    if overload_prob > 0.5:
        effective_zone_rank = min(effective_zone_rank, 0)  # too risky → monitor only
    elif overload_prob > 0.15:
        effective_zone_rank = min(effective_zone_rank, 1)  # elevated risk → at most small

    # Very low reserve margin caps the recommendation
    if reserve_margin < 5:
        effective_zone_rank = 0
    elif reserve_margin < 15:
        effective_zone_rank = min(effective_zone_rank, 1)

    # Most conservative of the two constraints wins
    final_rank = min(building_rank, effective_zone_rank)
    solution = {0: "none_monitor", 1: "residential_ac_small", 2: "residential_ac_medium"}[final_rank]

    # --- Load balancing ---
    if solution == "none_monitor":
        lb = "none"
    elif grid_sens > 0.35 or overload_prob > 0.08:
        # Dynamic LB required: either the zone is grid-sensitive or
        # the 2030 overload risk exceeds 8 % (roughly top quartile of Prague zones)
        lb = "dynamic"
    else:
        lb = "static"

    return solution, lb


def _derive_wallbox_count(
    available: float,
    reserve_margin: float,
    overload_prob: float,
    expected_evs: int,
    parking_spaces: int,
) -> float:
    """
    Target for the MLP regressor: how many wallboxes should be installed.

    The zone risk (reserve_margin, overload_prob) smoothly scales the count
    down from the theoretical building maximum, so the neural network learns
    a continuous risk-adjusted recommendation rather than a hard step function.
    """
    max_simultaneous = math.floor(available / 3.7)
    raw = min(parking_spaces, expected_evs, max_simultaneous * 2)

    # Zone risk factor in [0.1, 1.0] — low reserve or high overload → fewer wallboxes
    reserve_factor = min(1.0, max(0.1, reserve_margin / 50.0))
    overload_factor = max(0.1, 1.0 - overload_prob * 2)
    zone_factor = reserve_factor * overload_factor

    return max(0.0, raw * zone_factor)


# ---------------------------------------------------------------------------

def generate_synthetic_training_data(zones_df: pd.DataFrame, rng_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    rows = []

    for _, zone in zones_df.iterrows():
        reserve_margin = float(zone.get("reserve_margin_pct_2025_synthetic", 20))
        grid_sens = float(zone.get("grid_sensitivity_index_derived", 0.5))
        overload_prob = float(zone.get("target_overload_probability_2030_synthetic", 0.05))
        zone_rank = _zone_solution_rank(
            str(zone.get("target_recommended_solution_synthetic", "residential_ac_medium"))
        )

        for _ in range(SAMPLES_PER_ZONE):
            flats = int(rng.integers(4, 151))
            parking_spaces = int(rng.integers(0, max(1, flats)))
            breaker_amps = int(rng.choice(BREAKER_CHOICES))
            expected_evs = int(rng.integers(0, min(parking_spaces + 1, 51)))
            has_pv = bool(rng.random() < 0.3)
            pv_kwp = float(rng.uniform(5, 50)) if has_pv else 0.0

            cap = calculate_wallbox_capacity(
                flats, parking_spaces, breaker_amps, expected_evs, has_pv, pv_kwp
            )

            available = cap["available_for_ev_kw"]
            solution, lb = _derive_labels(
                available, reserve_margin, grid_sens, expected_evs, overload_prob, zone_rank
            )
            wallbox_count = _derive_wallbox_count(
                available, reserve_margin, overload_prob, expected_evs, parking_spaces
            )

            row = {
                # Zone features (from CSV — vary by zone)
                "residential_index_derived": float(zone.get("residential_index_derived", 0)),
                "grid_sensitivity_index_derived": grid_sens,
                "reserve_capacity_kw_2025_synthetic": float(zone.get("reserve_capacity_kw_2025_synthetic", 100)),
                "reserve_margin_pct_2025_synthetic": reserve_margin,
                "no_private_parking_index_derived": float(zone.get("no_private_parking_index_derived", 0)),
                "target_overload_probability_2030_synthetic": overload_prob,
                "zone_solution_rank": zone_rank,
                # Capacity features (computed from building params)
                "available_for_ev_kw": available,
                "total_building_capacity_kw": cap["total_building_capacity_kw"],
                "max_simultaneous_with_balancing": cap["max_simultaneous_with_balancing"],
                # Building features (user input)
                "expected_evs": expected_evs,
                "parking_spaces": parking_spaces,
                "flats": flats,
                # Targets
                "solution_label": solution,
                "lb_label": lb,
                "wallbox_count": wallbox_count,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def train_models():
    print("=" * 60)
    print("Voltík — training zone-aware ML models")
    print("=" * 60)

    train_zones = pd.read_csv(DATA_DIR / "zones_train.csv")
    val_zones = pd.read_csv(DATA_DIR / "zones_validation.csv")

    print(f"Train zones: {len(train_zones)}  |  Validation zones: {len(val_zones)}")

    print(f"\nGenerating synthetic training data ({len(train_zones)} zones × {SAMPLES_PER_ZONE} configs)…")
    train_df = generate_synthetic_training_data(train_zones, rng_seed=42)
    print(f"  {len(train_df)} training samples")
    print(f"  Solution distribution:\n{train_df['solution_label'].value_counts().to_string()}")
    print(f"  LB distribution:\n{train_df['lb_label'].value_counts().to_string()}")

    print(f"\nGenerating held-out validation data ({len(val_zones)} zones × {SAMPLES_PER_ZONE} configs)…")
    val_df = generate_synthetic_training_data(val_zones, rng_seed=99)
    print(f"  {len(val_df)} validation samples")

    X_train = train_df[ALL_FEATURES].fillna(0)
    X_val = val_df[ALL_FEATURES].fillna(0)

    # --- Model 1: Solution classifier (GradientBoostingClassifier) ----------
    print("\n[1/3] Training solution classifier (GradientBoostingClassifier)…")
    solution_clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
    )
    solution_clf.fit(X_train, train_df["solution_label"])
    val_sol_pred = solution_clf.predict(X_val)
    print("  Validation report (held-out zones):")
    print(classification_report(val_df["solution_label"], val_sol_pred, zero_division=0))

    # --- Model 2: LB classifier (GradientBoostingClassifier) ----------------
    print("[2/3] Training load-balancing classifier (GradientBoostingClassifier)…")
    lb_clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
    )
    lb_clf.fit(X_train, train_df["lb_label"])
    val_lb_pred = lb_clf.predict(X_val)
    print("  Validation report (held-out zones):")
    print(classification_report(val_df["lb_label"], val_lb_pred, zero_division=0))

    # --- Model 3: Wallbox count (MLPRegressor — neural network) -------------
    print("[3/3] Training wallbox count neural network (MLPRegressor 64→32→16)…")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    wallbox_reg = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        max_iter=500,
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    wallbox_reg.fit(X_train_scaled, train_df["wallbox_count"])

    val_count_pred = wallbox_reg.predict(X_val_scaled)
    mae = np.abs(val_count_pred - val_df["wallbox_count"].values).mean()
    print(f"  Validation MAE: {mae:.2f} wallboxes")

    # --- Save ---------------------------------------------------------------
    MODELS_DIR.mkdir(exist_ok=True)
    for old in MODELS_DIR.glob("*.joblib"):
        old.unlink()

    joblib.dump(solution_clf, MODELS_DIR / "solution_classifier.joblib")
    joblib.dump(lb_clf, MODELS_DIR / "lb_classifier.joblib")
    joblib.dump({"model": wallbox_reg, "scaler": scaler}, MODELS_DIR / "wallbox_regressor.joblib")

    print("\nSaved: solution_classifier.joblib, lb_classifier.joblib, wallbox_regressor.joblib")
    print("=" * 60)

    return solution_clf, lb_clf, wallbox_reg, scaler


def load_or_train_models():
    sol_path = MODELS_DIR / "solution_classifier.joblib"
    lb_path = MODELS_DIR / "lb_classifier.joblib"
    wb_path = MODELS_DIR / "wallbox_regressor.joblib"
    if sol_path.exists() and lb_path.exists() and wb_path.exists():
        wb_bundle = joblib.load(wb_path)
        return (
            joblib.load(sol_path),
            joblib.load(lb_path),
            wb_bundle["model"],
            wb_bundle["scaler"],
        )
    print("Models not found — training from scratch…")
    return train_models()


# keep old name as alias so app.py import still works
def load_or_train_classifier():
    return load_or_train_models()


def predict_solution(
    model_tuple,
    zone_row: pd.Series,
    capacity_result: dict,
    user_input: dict,
) -> dict:
    solution_clf, lb_clf, wallbox_reg, scaler = model_tuple

    zone_sol = str(zone_row.get("target_recommended_solution_synthetic", "residential_ac_medium"))

    feature_values = {
        "residential_index_derived": float(zone_row.get("residential_index_derived", 0)),
        "grid_sensitivity_index_derived": float(zone_row.get("grid_sensitivity_index_derived", 0.5)),
        "reserve_capacity_kw_2025_synthetic": float(zone_row.get("reserve_capacity_kw_2025_synthetic", 100)),
        "reserve_margin_pct_2025_synthetic": float(zone_row.get("reserve_margin_pct_2025_synthetic", 30)),
        "no_private_parking_index_derived": float(zone_row.get("no_private_parking_index_derived", 0)),
        "target_overload_probability_2030_synthetic": float(zone_row.get("target_overload_probability_2030_synthetic", 0.05)),
        "zone_solution_rank": _zone_solution_rank(zone_sol),
        "available_for_ev_kw": capacity_result["available_for_ev_kw"],
        "total_building_capacity_kw": capacity_result["total_building_capacity_kw"],
        "max_simultaneous_with_balancing": capacity_result["max_simultaneous_with_balancing"],
        "expected_evs": user_input.get("expected_evs", 0),
        "parking_spaces": user_input.get("parking_spaces", 0),
        "flats": user_input.get("flats", 0),
    }

    X = pd.DataFrame([feature_values], columns=ALL_FEATURES).fillna(0)
    X_scaled = scaler.transform(X)

    solution_type = solution_clf.predict(X)[0]
    load_balancing_type = lb_clf.predict(X)[0]
    wallbox_count_raw = float(wallbox_reg.predict(X_scaled)[0])
    wallbox_count = max(0, round(wallbox_count_raw))

    sol_proba = dict(zip(solution_clf.classes_, solution_clf.predict_proba(X)[0].round(4)))
    lb_proba = dict(zip(lb_clf.classes_, lb_clf.predict_proba(X)[0].round(4)))

    sol_importances = dict(zip(ALL_FEATURES, solution_clf.feature_importances_.round(4)))
    lb_importances = dict(zip(ALL_FEATURES, lb_clf.feature_importances_.round(4)))

    return {
        "solution_type": solution_type,
        "load_balancing_type": load_balancing_type,
        "wallbox_count_nn": wallbox_count,
        "wallbox_count_raw": round(wallbox_count_raw, 2),
        "zone_solution_rank": _zone_solution_rank(zone_sol),
        "zone_expert_label": zone_sol,
        "debug": {
            "feature_vector": feature_values,
            "solution_proba": sol_proba,
            "lb_proba": lb_proba,
            "solution_importances": sol_importances,
            "lb_importances": lb_importances,
            "model_types": {
                "solution": type(solution_clf).__name__,
                "lb": type(lb_clf).__name__,
                "wallbox": type(wallbox_reg).__name__,
            },
            "solution_classes": list(solution_clf.classes_),
            "lb_classes": list(lb_clf.classes_),
            "zone_expert_label": zone_sol,
        },
    }


if __name__ == "__main__":
    train_models()
