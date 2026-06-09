import math


WALLBOX_POWER_KW = 11.0
MIN_DYNAMIC_POWER_KW = 3.7


def breaker_to_kw(amps: float) -> float:
    return math.sqrt(3) * 400 * amps * 0.95 / 1000


def _diversity_factor(flats: int) -> float:
    """
    Coincidence/diversity factor per Czech electrical engineering practice (ČSN 33 2130).
    Not all flats draw peak power simultaneously; factor drops with building size.
    """
    if flats <= 4:
        return 0.70
    if flats <= 8:
        return 0.55
    if flats <= 15:
        return 0.42
    if flats <= 25:
        return 0.32
    if flats <= 40:
        return 0.25
    if flats <= 70:
        return 0.20
    if flats <= 120:
        return 0.17
    return 0.14


def estimate_building_base_load_kw(flats: int) -> dict:
    """
    Diversity-factored building load.
    Assumes 7 kW installed per flat (standard Czech 32A single-phase breaker).
    Evening peak applies the coincidence factor; daytime base is ~65 % of that.
    """
    df = _diversity_factor(flats)
    # 5 kW per flat reflects the typical Czech residential flat (single-phase 25A breaker)
    per_flat_kw = 5.0
    evening_peak_kw = flats * per_flat_kw * df
    base_kw = evening_peak_kw * 0.65
    return {
        "base_kw": round(base_kw, 1),
        "evening_peak_kw": round(evening_peak_kw, 1),
    }


def calculate_wallbox_capacity(
    flats: int,
    parking_spaces: int,
    breaker_amps: float,
    expected_evs: int,
    has_pv: bool,
    pv_kwp: float,
) -> dict:
    total_building_capacity_kw = breaker_to_kw(breaker_amps)
    base = estimate_building_base_load_kw(flats)
    estimated_evening_base_load_kw = base["evening_peak_kw"]

    available_for_ev_kw = max(0.0, total_building_capacity_kw - estimated_evening_base_load_kw)

    max_wallboxes_without_balancing = math.floor(available_for_ev_kw / WALLBOX_POWER_KW)
    max_simultaneous_with_balancing = math.floor(available_for_ev_kw / MIN_DYNAMIC_POWER_KW)

    # How many physical wallboxes make sense to install
    # (dynamic LB means more wallboxes can share capacity than charge simultaneously)
    recommended_installed_wallboxes = min(
        parking_spaces,
        expected_evs,
        max(0, max_simultaneous_with_balancing * 2),
    )

    pv_peak_kw = 0.0
    if has_pv and pv_kwp > 0:
        pv_peak_kw = pv_kwp * 0.75  # daytime surplus — not counted for evening guaranteed capacity

    if available_for_ev_kw < 7:
        warning_level = "red"
    elif max_wallboxes_without_balancing == 0 or max_simultaneous_with_balancing < 2:
        warning_level = "orange"
    else:
        warning_level = "green"

    return {
        "total_building_capacity_kw": round(total_building_capacity_kw, 1),
        "estimated_evening_base_load_kw": round(estimated_evening_base_load_kw, 1),
        "available_for_ev_kw": round(available_for_ev_kw, 1),
        "max_wallboxes_without_balancing": max_wallboxes_without_balancing,
        "max_simultaneous_with_balancing": max_simultaneous_with_balancing,
        "recommended_installed_wallboxes": recommended_installed_wallboxes,
        "pv_peak_kw": round(pv_peak_kw, 1),
        "warning_level": warning_level,
    }
