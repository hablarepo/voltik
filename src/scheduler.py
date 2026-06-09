import numpy as np
import pandas as pd

# Normalised solar irradiance curve (hour -> fraction of peak)
SOLAR_CURVE = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
    6: 0.05, 7: 0.15, 8: 0.35, 9: 0.60, 10: 0.80,
    11: 0.95, 12: 1.00, 13: 0.98, 14: 0.90, 15: 0.75,
    16: 0.50, 17: 0.25, 18: 0.10, 19: 0.02, 20: 0,
    21: 0, 22: 0, 23: 0,
}

PEAK_HOURS = {18, 19, 20, 21}
PEAK_PENALTY = 100


def create_hourly_charging_plan(
    grid_zone_id: str,
    expected_daily_kwh: float,
    has_pv: bool,
    pv_kwp: float,
    hourly_agg: pd.DataFrame,
    available_for_ev_kw: float,
) -> pd.DataFrame:
    zone_data = hourly_agg[hourly_agg["grid_zone_id"] == grid_zone_id].copy()

    if zone_data.empty:
        # Fallback: flat profile when no zone data
        zone_data = pd.DataFrame({
            "hour": range(24),
            "avg_available_kw": [available_for_ev_kw] * 24,
            "overload_rate": [0.0] * 24,
        })
    else:
        zone_data = zone_data[["hour", "avg_available_kw", "overload_rate"]].copy()

    zone_data = zone_data.set_index("hour").reindex(range(24)).fillna(
        {"avg_available_kw": available_for_ev_kw, "overload_rate": 0}
    )

    scores = zone_data["avg_available_kw"] - PEAK_PENALTY * zone_data["overload_rate"]

    pv_gen = pd.Series(0.0, index=range(24))
    if has_pv and pv_kwp > 0:
        for h in range(24):
            pv_gen[h] = pv_kwp * SOLAR_CURVE.get(h, 0)
        scores += pv_gen * 2

    for h in PEAK_HOURS:
        scores[h] -= PEAK_PENALTY

    # Allocate daily kWh greedily across best-scoring hours
    max_per_hour = zone_data["avg_available_kw"].clip(upper=available_for_ev_kw)
    max_per_hour = max_per_hour.clip(lower=0)

    remaining = expected_daily_kwh
    allocated = pd.Series(0.0, index=range(24))
    for h in scores.sort_values(ascending=False).index:
        if remaining <= 0:
            break
        capacity = float(max_per_hour.get(h, available_for_ev_kw))
        charge = min(capacity, remaining)
        allocated[h] = round(charge, 2)
        remaining -= charge

    reasons = []
    for h in range(24):
        if h in PEAK_HOURS:
            reasons.append("Špička sítě – nenabíjet")
        elif has_pv and 9 <= h <= 16 and pv_gen[h] > 0.5:
            reasons.append("Přebytek FVE – ideální čas")
        elif scores[h] >= scores.quantile(0.7):
            reasons.append("Dobrá dostupnost sítě")
        elif allocated[h] > 0:
            reasons.append("Doplňkový čas")
        else:
            reasons.append("")

    plan = pd.DataFrame({
        "hour": range(24),
        "grid_score": scores.values.round(1),
        "pv_generation_kw": pv_gen.values.round(2),
        "recommended_charging_kw": allocated.values,
        "reason": reasons,
    })
    return plan
