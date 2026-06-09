import math


WALLBOX_POWER_KW = 11.0
MIN_DYNAMIC_POWER_KW = 3.7


def breaker_to_kw(amps: float) -> float:
    return math.sqrt(3) * 400 * amps * 0.95 / 1000


def estimate_building_base_load_kw(flats: int) -> dict:
    return {
        "base_kw": flats * 0.8,
        "evening_peak_kw": flats * 1.4,
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
    recommended_installed_wallboxes = min(
        parking_spaces,
        expected_evs,
        max_simultaneous_with_balancing * 2,
    )
    recommended_installed_wallboxes = max(0, recommended_installed_wallboxes)

    pv_peak_kw = 0.0
    if has_pv and pv_kwp > 0:
        pv_peak_kw = pv_kwp * 0.75  # daytime only — not counted for evening guaranteed capacity

    # Warning level
    if available_for_ev_kw < 7:
        warning_level = "red"
    elif max_wallboxes_without_balancing == 0 or max_simultaneous_with_balancing < 2:
        warning_level = "orange"
    elif max_wallboxes_without_balancing < 2:
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
