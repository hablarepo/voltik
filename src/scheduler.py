import numpy as np
import pandas as pd

# Solar irradiance fraction of peak by hour
SOLAR_CURVE = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0.02,
    6: 0.08, 7: 0.22, 8: 0.45, 9: 0.65, 10: 0.82,
    11: 0.93, 12: 1.00, 13: 0.97, 14: 0.87, 15: 0.70,
    16: 0.48, 17: 0.25, 18: 0.08, 19: 0.01, 20: 0,
    21: 0, 22: 0, 23: 0,
}

# Practical resident charging priority by hour
# Night (22-06): cars parked, low grid load → best
# Morning (06-09): before work rush → good
# Midday (10-15): PV bonus zone → good if PV
# Afternoon (15-18): moderate
# Evening peak (18-21): grid stressed → avoid
HOUR_BASE_SCORE = {
    0: 85, 1: 90, 2: 90, 3: 90, 4: 90, 5: 85,
    6: 70, 7: 60, 8: 55, 9: 50,
    10: 45, 11: 45, 12: 45, 13: 45, 14: 45,
    15: 35, 16: 30, 17: 20,
    18: -30, 19: -40, 20: -40, 21: -30,
    22: 75, 23: 80,
}

PEAK_HOURS = {18, 19, 20, 21}


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
        zone_data = pd.DataFrame({
            "hour": range(24),
            "avg_available_kw": [available_for_ev_kw] * 24,
            "overload_rate": [0.0] * 24,
        })
    else:
        zone_data = zone_data[["hour", "avg_available_kw", "overload_rate"]].copy()

    zone_data = (
        zone_data.set_index("hour")
        .reindex(range(24))
        .fillna({"avg_available_kw": available_for_ev_kw, "overload_rate": 0})
    )

    # Composite score: practical priority + grid health penalty for overload risk
    scores = pd.Series(
        {h: HOUR_BASE_SCORE[h] for h in range(24)},
        dtype=float,
    )

    # Reduce score for hours with historically high overload risk in this zone
    overload_penalty = zone_data["overload_rate"] * 150
    scores -= overload_penalty

    # PV bonus: midday hours become much more attractive when solar available
    pv_gen = pd.Series(0.0, index=range(24))
    if has_pv and pv_kwp > 0:
        for h in range(24):
            pv_gen[h] = pv_kwp * SOLAR_CURVE.get(h, 0)
        # PV bonus: at 10+ kWp midday should beat night tariff hours
        scores += pv_gen * 5

    # Grid capacity cap: don't exceed what the zone and building can handle
    grid_cap = zone_data["avg_available_kw"].clip(upper=available_for_ev_kw)
    grid_cap = grid_cap.clip(lower=0)

    # Hard block peak hours — never charge 18–21
    for h in PEAK_HOURS:
        scores[h] = -999
        grid_cap[h] = 0

    # Allocate daily kWh greedily in score order
    remaining = max(0.0, expected_daily_kwh)
    allocated = pd.Series(0.0, index=range(24))

    for h in scores.sort_values(ascending=False).index:
        if remaining <= 0:
            break
        cap = float(grid_cap.get(h, 0))
        if cap <= 0:
            continue
        charge = min(cap, remaining)
        allocated[h] = round(charge, 2)
        remaining -= charge

    # Human-readable reasons
    reasons = []
    for h in range(24):
        if h in PEAK_HOURS:
            reasons.append("Síťová špička – nenabíjet")
        elif allocated[h] > 0 and has_pv and pv_gen[h] > 1.0:
            reasons.append("Přebytek FVE – ideální čas")
        elif allocated[h] > 0 and (h >= 22 or h <= 5):
            reasons.append("Noční tarif – nízká cena")
        elif allocated[h] > 0 and 6 <= h <= 9:
            reasons.append("Ranní doplnění")
        elif allocated[h] > 0:
            reasons.append("Dostupná kapacita sítě")
        else:
            reasons.append("")

    return pd.DataFrame({
        "hour": range(24),
        "grid_score": scores.values.round(1),
        "pv_generation_kw": pv_gen.values.round(2),
        "recommended_charging_kw": allocated.values,
        "reason": reasons,
    })
