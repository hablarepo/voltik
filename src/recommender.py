import pandas as pd


SOLUTION_LABELS = {
    "none_monitor": "Zatím bez wallboxů – monitoring",
    "residential_ac_small": "Malá rezidenční AC stanice (2× 11 kW)",
    "residential_ac_medium": "Střední rezidenční AC stanice (6× 11 kW)",
    "destination_dc50": "Rychlonabíjení DC 50 kW",
    "fast_hub_150": "Rychlý hub 150 kW",
    "mixed_mobility_hub": "Smíšený mobility hub AC/DC",
}

BALANCING_LABELS = {
    "none": "Bez load balancingu",
    "static": "Statický load balancing",
    "dynamic": "Dynamický load balancing",
}


def recommend_configuration(
    user_input: dict,
    zone_row: pd.Series,
    capacity_result: dict,
    ml_prediction: dict,
) -> dict:
    available_for_ev_kw = capacity_result["available_for_ev_kw"]
    reserve_margin = float(zone_row.get("reserve_margin_pct_2025_synthetic", 20))
    grid_sensitivity = float(zone_row.get("grid_sensitivity_index_derived", 0.5))
    expected_evs = user_input.get("expected_evs", 0)
    wallboxes = capacity_result["recommended_installed_wallboxes"]

    # Use ML predictions as the starting point
    solution_type = ml_prediction.get("solution_type", "none_monitor")
    load_balancing = ml_prediction.get("load_balancing_type", "static")
    warnings = []
    risk_score = 0

    # Safety overrides
    if available_for_ev_kw < 7:
        solution_type = "none_monitor"
        load_balancing = "none"
        warnings.append("Dostupný výkon pro EV je příliš nízký. Doporučujeme pouze monitoring.")
        risk_score += 40

    if reserve_margin < 10:
        warnings.append("Rezerva distribuční sítě je velmi nízká – pravděpodobně bude nutná úprava přípojky.")
        risk_score += 30

    if grid_sensitivity > 0.75:
        load_balancing = "dynamic"
        risk_score += 20

    if solution_type == "none_monitor":
        load_balancing = "none"
        wallboxes = 0

    risk_score = min(100, risk_score)

    explanation_parts = [
        f"Na základě parametrů vašeho domu a dat pražské distribuční sítě (zóna {zone_row['grid_zone_id']}) "
        f"doporučujeme řešení typu **{SOLUTION_LABELS.get(solution_type, solution_type)}**.",
    ]
    if load_balancing == "dynamic":
        explanation_parts.append(
            "Kvůli vysoké citlivosti sítě v dané zóně doporučujeme **dynamický load balancing**, "
            "který průběžně přizpůsobuje výkon wallboxů aktuálnímu zatížení."
        )
    elif load_balancing == "static":
        explanation_parts.append(
            "Doporučujeme **statický load balancing** – pevné rozdělení výkonu mezi wallboxy."
        )
    for w in warnings:
        explanation_parts.append(f"⚠️ {w}")

    explanation_parts.append(
        "Voltík není náhrada za elektroprojektanta. Jde o rychlý AI pre-assessment, "
        "který SVJ ukáže, zda je projekt pravděpodobně realizovatelný a jakou konfiguraci řešit."
    )

    recommended_total_kw = wallboxes * 11.0 if solution_type != "none_monitor" else 0.0

    return {
        "recommended_solution_type": solution_type,
        "solution_label": SOLUTION_LABELS.get(solution_type, solution_type),
        "number_of_wallboxes": wallboxes,
        "recommended_total_kw": recommended_total_kw,
        "load_balancing_type": load_balancing,
        "load_balancing_label": BALANCING_LABELS.get(load_balancing, load_balancing),
        "explanation": "\n\n".join(explanation_parts),
        "risk_score": risk_score,
    }
