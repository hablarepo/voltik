import os
import streamlit as st


def _get_client():
    """Returns Mistral client or None if not configured."""
    try:
        from mistralai.client import Mistral
    except ImportError:
        try:
            from mistralai import Mistral  # older SDK layout
        except ImportError:
            return None
    try:
        key = os.environ.get("MISTRAL_API_KEY") or st.secrets.get("MISTRAL_API_KEY", "")
        if not key:
            return None
        return Mistral(api_key=key)
    except Exception:
        return None


def generate_explanation(
    recommendation: dict,
    capacity: dict,
    zone_id: str,
    zone_row,  # pd.Series
    user_input: dict,
    economics: dict,
) -> str:
    """Generate Czech explanation via Mistral. Falls back to template."""
    client = _get_client()
    if client is None:
        return recommendation.get("explanation", "")

    prompt = f"""Jsi AI poradce Voltík pro správce bytových domů (SVJ) v Praze. Vysvětli doporučení pro instalaci wallboxů pro elektromobily.

Parametry domu:
- Bytové jednotky: {user_input['flats']}
- Parkovací místa: {user_input['parking_spaces']}
- Hlavní jistič: 3×{user_input['breaker_amps']} A
- Očekávané počty EV: {user_input['expected_evs']}
- FVE: {'Ano, ' + str(user_input.get('pv_kwp', 0)) + ' kWp' if user_input.get('has_pv') else 'Ne'}
- Distribuční zóna: {zone_id}

Kapacitní analýza:
- Celková kapacita přípojky: {capacity['total_building_capacity_kw']} kW
- Dostupný výkon pro EV: {capacity['available_for_ev_kw']} kW
- Max wallboxy bez load balancingu: {capacity['max_wallboxes_without_balancing']}
- Max wallboxy s dynamickým LB: {capacity['max_simultaneous_with_balancing']}

Doporučení:
- Řešení: {recommendation['solution_label']}
- Počet wallboxů: {recommendation['number_of_wallboxes']}
- Load balancing: {recommendation['load_balancing_label']}
- Skóre rizika: {recommendation['risk_score']}/100
- Rezerva sítě v zóně: {zone_row.get('reserve_margin_pct_2025_synthetic', 0):.1f}%

Ekonomika:
- Celkové náklady: {economics['total_cost']:,.0f} Kč
- Měsíční úspora: {economics['monthly_savings']:,.0f} Kč
- Návratnost: {economics['payback_years']:.1f} let

Napiš stručné, profesionální vysvětlení v češtině (4-6 vět) pro členy SVJ. Zdůrazni klíčová rizika, výhody a co dál podniknout. Zakončení: připomeň, že Voltík je pre-assessment a finální projekt musí vypracovat certifikovaný elektroprojektant."""

    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception:
        return recommendation.get("explanation", "")


def chat_response(
    user_message: str,
    recommendation: dict,
    capacity: dict,
    zone_id: str,
    user_input: dict,
) -> str:
    """Answer a follow-up question about the EV installation. Falls back to error message."""
    client = _get_client()
    if client is None:
        return "Chatbot není dostupný — nastavte proměnnou prostředí MISTRAL_API_KEY."

    system = f"""Jsi Voltík, AI poradce pro SVJ v Praze ohledně instalace wallboxů pro EV.
Odpovídej stručně a věcně v češtině.

Kontext domu: {user_input['flats']} bytů, jistič 3×{user_input['breaker_amps']}A,
{user_input['expected_evs']} EV, zóna {zone_id}.
Doporučení: {recommendation['solution_label']}, {recommendation['number_of_wallboxes']} wallboxů,
{recommendation['load_balancing_label']}.
Dostupný výkon: {capacity['available_for_ev_kw']} kW."""

    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Chyba při generování odpovědi: {e}"
