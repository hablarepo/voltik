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
) -> dict:
    home_price_czk_kwh = 6
    public_price_czk_kwh = 12
    avg_monthly_kwh_per_ev = 250

    monthly_savings = expected_evs * avg_monthly_kwh_per_ev * (public_price_czk_kwh - home_price_czk_kwh)
    annual_savings = monthly_savings * 12

    base_cost = 80_000
    wallbox_cost = number_of_wallboxes * 25_000
    lb_cost = 60_000 if load_balancing_type == "dynamic" else 0
    pv_cost = 40_000 if has_pv else 0
    total_cost = base_cost + wallbox_cost + lb_cost + pv_cost

    payback_years = total_cost / annual_savings if annual_savings > 0 else float("inf")

    return {
        "monthly_savings": monthly_savings,
        "annual_savings": annual_savings,
        "base_cost": base_cost,
        "wallbox_cost": wallbox_cost,
        "lb_cost": lb_cost,
        "pv_cost": pv_cost,
        "total_cost": total_cost,
        "payback_years": payback_years,
    }
