def risk_color(score: int) -> str:
    if score >= 60:
        return "#e74c3c"
    if score >= 30:
        return "#f39c12"
    return "#27ae60"


def warning_color(level: str) -> str:
    return {"green": "#27ae60", "orange": "#f39c12", "red": "#e74c3c"}.get(level, "#888")


def warning_emoji(level: str) -> str:
    return {"green": "✅", "orange": "⚠️", "red": "🔴"}.get(level, "")


def calculate_economics(
    expected_evs: int,
    number_of_wallboxes: int,
    has_pv: bool,
    load_balancing_type: str,
    pv_kwh_daily: float = 0.0,
    expected_daily_kwh: float = 0.0,
) -> dict:
    home_price_czk_kwh = 6
    public_price_czk_kwh = 12
    avg_monthly_kwh_per_ev = 300

    # Fraction of EV charging supplied by solar
    pv_fraction = 0.0
    if has_pv and pv_kwh_daily > 0 and expected_daily_kwh > 0:
        pv_fraction = min(1.0, pv_kwh_daily / expected_daily_kwh)

    monthly_kwh_total = expected_evs * avg_monthly_kwh_per_ev
    monthly_kwh_pv = monthly_kwh_total * pv_fraction      # effectively free
    monthly_kwh_grid = monthly_kwh_total - monthly_kwh_pv  # paid at home tariff

    # Grid kWh: save (public − home) per kWh vs. public charging
    # PV kWh: save the entire public price (costs nothing from solar)
    monthly_savings = (
        monthly_kwh_grid * (public_price_czk_kwh - home_price_czk_kwh)
        + monthly_kwh_pv * public_price_czk_kwh
    )
    annual_savings = monthly_savings * 12

    base_cost = 80_000
    wallbox_cost = number_of_wallboxes * 25_000
    lb_cost = 60_000 if load_balancing_type == "dynamic" else 0
    pv_cost = 40_000 if has_pv else 0   # smart-charger PV integration
    total_cost = base_cost + wallbox_cost + lb_cost + pv_cost

    payback_years = total_cost / annual_savings if annual_savings > 0 else float("inf")

    return {
        "monthly_savings": round(monthly_savings),
        "annual_savings": round(annual_savings),
        "base_cost": base_cost,
        "wallbox_cost": wallbox_cost,
        "lb_cost": lb_cost,
        "pv_cost": pv_cost,
        "total_cost": total_cost,
        "payback_years": payback_years,
        "pv_fraction": round(pv_fraction, 3),
        "monthly_kwh_pv": round(monthly_kwh_pv),
        "monthly_kwh_grid": round(monthly_kwh_grid),
    }
