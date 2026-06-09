import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os
import random
import time
import uuid
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data_loader import (
    load_zones, load_hourly_aggregated, load_grid_capacity,
    load_candidate_solutions, get_zone_row,
)
from src.capacity_calculator import calculate_wallbox_capacity, breaker_to_kw
from src.train_models import load_or_train_classifier, predict_solution
from src.recommender import recommend_configuration
from src.scheduler import create_hourly_charging_plan, PEAK_HOURS
from src.document_generator import generate_svj_document
from src.utils import warning_color, warning_emoji, calculate_economics, risk_color
from src.ai_explainer import generate_explanation, chat_response

st.set_page_config(
    page_title="Voltík – EV poradce pro SVJ",
    page_icon="⚡",
    layout="wide",
)

# ── Static CSS (layout, colours — no animations here) ─────────────────────────
st.markdown("""
<style>
.bar-wrap {
    height: 22px; border-radius: 10px; background: #e9ecef;
    width: 100%; margin: 8px 0 14px 0; position: relative; overflow: hidden;
}
.bar-legend {
    display: flex; gap: 20px; font-size: 0.8rem; color: #6c757d; margin-bottom: 6px;
}
.mc-wrap { display: flex; gap: 12px; margin-bottom: 4px; }
.mc {
    flex: 1; background: #f4f6fb; border-radius: 10px;
    padding: 14px 18px; border: 1px solid #dee2e6;
}
.mc-label { font-size: 0.78rem; color: #6c757d; margin-bottom: 4px; }
.mc-value { font-size: 1.65rem; font-weight: 700; color: #1a202c; line-height: 1.2; display: inline-block; }
.mc-delta { font-size: 0.76rem; color: #6c757d; margin-top: 3px; }
.recommendation-box {
    background: #f0f7ff; border-radius: 12px;
    padding: 20px 24px; border: 1px solid #bdd7f5; margin-top: 12px;
}
.rec-table { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
.rec-table td { padding: 7px 0; }
.rec-table tr { border-bottom: 1px solid #d0e4f7; }
.rec-table tr:last-child { border-bottom: none; }
.rec-label { color: #6c757d; }
.rec-val { font-weight: 700; text-align: right; }
.rec-val-big { font-weight: 700; text-align: right; font-size: 1.4rem; display: inline-block; }
.rec-note { font-size: 0.8rem; color: #6c757d; margin-top: 10px; line-height: 1.5; }
.risk-bar-wrap { background: #e9ecef; border-radius: 8px; height: 14px; margin: 0 20px; overflow: hidden; }

/* ── Dim overlay while spinner is active ── */
@keyframes dimIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
body:has([data-testid="stSpinner"])::before {
    content: '';
    position: fixed;
    inset: 0;
    background: rgba(248, 249, 250, 0.80);
    backdrop-filter: blur(2px);
    z-index: 9998;
    animation: dimIn 0.2s ease both;
    pointer-events: all;
}
/* Spinner floats as a centred modal card above the dim */
[data-testid="stSpinner"] {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    z-index: 9999 !important;
    background: #ffffff !important;
    padding: 24px 40px !important;
    border-radius: 16px !important;
    box-shadow: 0 12px 48px rgba(0,0,0,0.13) !important;
    min-width: 260px !important;
    text-align: center !important;
}

/* AI button — gradient primary */
button[kind="primaryFormSubmit"],
button[kind="primary"] {
    background: linear-gradient(135deg, #2c7be5 0%, #6610f2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 14px rgba(44,123,229,0.35) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 20px rgba(44,123,229,0.50) !important;
}
button[kind="primary"]:active { transform: translateY(0) !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ Voltík — AI poradce pro nabíjení v bytovém domě")
st.caption(
    "Voltík kombinuje parametry vašeho domu se daty pražské distribuční sítě "
    "a vytváří rychlý AI pre-assessment toho, zda je projekt wallboxů reálný, "
    "jakou konfiguraci zvolit a jak se vyhnout přetížení sítě ve špičce."
)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Načítám data…"):
    zones_df = load_zones()
    hourly_agg = load_hourly_aggregated()
    solutions_df = load_candidate_solutions()
    classifier = load_or_train_classifier()

zone_ids = sorted(zones_df["grid_zone_id"].unique())

_good = zones_df[
    zones_df["target_recommended_solution_synthetic"] == "residential_ac_medium"
]["grid_zone_id"].unique()
_default_zone_idx = zone_ids.index(sorted(_good)[0]) if len(_good) > 0 else 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
BREAKER_OPTIONS = [25, 32, 40, 50, 63, 80, 100, 125, 160, 200, 250]

with st.sidebar:
    st.header("Parametry domu")

    flats = st.slider("Počet bytových jednotek", 4, 200, 24)
    parking_spaces = st.slider("Počet parkovacích míst", 0, 200, 12)
    breaker_amps = st.selectbox(
        "Hlavní jistič budovy",
        BREAKER_OPTIONS,
        index=6,
        format_func=lambda x: f"3×{x} A  ({breaker_to_kw(x):.0f} kW)",
    )
    expected_evs = st.slider("Očekávaný počet EV v domě", 0, 100, 5)

    st.divider()
    has_pv = st.checkbox("Střešní fotovoltaika (FVE)", value=False)
    pv_kwp = 0.0
    if has_pv:
        pv_kwp = st.number_input("Výkon FVE (kWp)", min_value=0.0, max_value=500.0,
                                  value=20.0, step=1.0)

    st.divider()
    selected_zone = st.selectbox("Distribuční zóna Praha", zone_ids, index=_default_zone_idx)
    scenario = st.selectbox(
        "Scénář adopce EV do 2030",
        ["conservative", "central", "ambitious"],
        index=1,
        format_func=lambda s: {
            "conservative": "Konzervativní",
            "central": "Střední",
            "ambitious": "Ambiciózní",
        }[s],
    )

    st.divider()
    st.caption(
        "Tip: pro větší domy (40+ bytů) použijte jistič 160 A nebo vyšší. "
        "Jistič zobrazuje celkový výkon přípojky budovy, nikoli jistič bytu."
    )

# ── Compute ───────────────────────────────────────────────────────────────────
user_input = {
    "flats": flats,
    "parking_spaces": parking_spaces,
    "breaker_amps": breaker_amps,
    "expected_evs": expected_evs,
    "has_pv": has_pv,
    "pv_kwp": pv_kwp,
    "scenario": scenario,
}

_input_key = str(user_input) + selected_zone
_params_changed = st.session_state.get("_last_input_key") != _input_key
if _params_changed:
    st.session_state.pop("ai_explanation", None)
    st.session_state["_last_input_key"] = _input_key

# Artificial "AI thinking" delay — makes the tool feel like it's doing real work
if _params_changed:
    with st.spinner("⚡ Voltík analyzuje data distribuční sítě…"):
        time.sleep(random.uniform(1.0, 2.0))

capacity = calculate_wallbox_capacity(
    flats, parking_spaces, breaker_amps, expected_evs, has_pv, pv_kwp
)
zone_row = get_zone_row(zones_df, selected_zone)

# Apply 2030 adoption scenario: scale overload probability and reserve margin.
# Conservative → less grid stress; Ambitious → more stress. This makes the
# scenario selector actually change the ML prediction and risk score.
_scenario_overload = {"conservative": 0.60, "central": 1.0, "ambitious": 1.55}
_scenario_reserve  = {"conservative": 1.18, "central": 1.0, "ambitious": 0.82}
zone_row_adj = zone_row.copy()
zone_row_adj["target_overload_probability_2030_synthetic"] = min(
    0.99,
    float(zone_row.get("target_overload_probability_2030_synthetic", 0.05))
    * _scenario_overload[scenario],
)
zone_row_adj["reserve_margin_pct_2025_synthetic"] = max(
    3.0,
    float(zone_row.get("reserve_margin_pct_2025_synthetic", 30))
    * _scenario_reserve[scenario],
)

ml_prediction = predict_solution(classifier, zone_row_adj, capacity, user_input)
recommendation = recommend_configuration(user_input, zone_row_adj, capacity, ml_prediction)

expected_daily_kwh = expected_evs * 12
plan_df = create_hourly_charging_plan(
    selected_zone, expected_daily_kwh, has_pv, pv_kwp,
    hourly_agg, capacity["available_for_ev_kw"],
)

pv_kwh_from_plan = (
    plan_df[(plan_df["hour"] >= 10) & (plan_df["hour"] <= 15)]["recommended_charging_kw"].sum()
    if has_pv else 0.0
)

economics = calculate_economics(
    expected_evs,
    recommendation["number_of_wallboxes"],
    has_pv,
    recommendation["load_balancing_type"],
    pv_kwh_daily=pv_kwh_from_plan,
    expected_daily_kwh=expected_daily_kwh,
)

# Unique render ID — forces new CSS animation names every rerender so
# browsers always re-fire the animation (CSS animations only trigger on
# element insertion; unique names make React treat elements as new).
_rid = uuid.uuid4().hex[:8]

# ── Capacity section ──────────────────────────────────────────────────────────
st.subheader("Kapacitní přehled")

wl = capacity["warning_level"]
wl_color = {"green": "#198754", "orange": "#fd7e14", "red": "#dc3545"}[wl]
wl_text = {
    "green": "Kapacita dostačující",
    "orange": "Nutný load balancing",
    "red": "Kapacita nedostačující",
}[wl]

total_kw = capacity["total_building_capacity_kw"]
base_kw  = capacity["estimated_evening_base_load_kw"]
avail_kw = capacity["available_for_ev_kw"]
simult   = capacity["max_simultaneous_with_balancing"]
base_pct  = min(100, base_kw  / total_kw * 100) if total_kw > 0 else 100
avail_pct = min(100 - base_pct, avail_kw / total_kw * 100) if total_kw > 0 else 0

st.markdown(
    f"**Přípojka budovy: {total_kw} kW** &nbsp;|&nbsp; "
    f"Odhadovaná večerní spotřeba: **{base_kw} kW** &nbsp;|&nbsp; "
    f"Volné pro EV: <span style='color:{wl_color}; font-weight:700'>{avail_kw} kW</span> "
    f"— {wl_text} {warning_emoji(wl)}",
    unsafe_allow_html=True,
)

# Animated capacity bar.
# clip-path: inset(0 100% 0 0) → inset(0 0 0 0) reveals the bar left-to-right
# without touching the `width` attribute (which avoids the !important hack that
# browsers silently discard inside @keyframes).
# UUID suffix forces a new animation-name each render → browser always re-fires.
st.markdown(f"""
<style>
@keyframes bf{_rid} {{
    from {{ clip-path: inset(0 100% 0 0); }}
    to   {{ clip-path: inset(0 0% 0 0); }}
}}
@keyframes pi{_rid} {{
    0%   {{ transform: translateY(8px) scale(0.88); opacity: 0; }}
    60%  {{ transform: translateY(-2px) scale(1.03); opacity: 1; }}
    100% {{ transform: translateY(0)    scale(1);    opacity: 1; }}
}}
@keyframes fu{_rid} {{
    from {{ transform: translateY(10px); opacity: 0; }}
    to   {{ transform: translateY(0);    opacity: 1; }}
}}
</style>
<div class="bar-wrap">
  <div style="position:absolute; left:0; width:{base_pct:.2f}%; height:100%;
              background:#dc3545; border-radius:10px 0 0 10px;
              animation: bf{_rid} 0.55s cubic-bezier(0.25,0.46,0.45,0.94) both"></div>
  <div style="position:absolute; left:{base_pct:.2f}%; width:{avail_pct:.2f}%; height:100%;
              background:{wl_color};
              animation: bf{_rid} 0.75s cubic-bezier(0.25,0.46,0.45,0.94) 0.08s both"></div>
</div>
<div class="bar-legend">
  <span><span style="color:#dc3545">■</span> Spotřeba domu ({base_kw} kW)</span>
  <span><span style="color:{wl_color}">■</span> Dostupné pro EV ({avail_kw} kW)</span>
  <span><span style="color:#ced4da">■</span> Rezerva jističe</span>
</div>
""", unsafe_allow_html=True)

# Animated metric cards — unique animation name per render.
# Wallbox counts without/with LB are capped at the recommended installation count
# so they never show a physically-possible number that exceeds what we actually
# advise installing (which would confuse the reader).
installed  = recommendation["number_of_wallboxes"]
wb_no_lb   = min(capacity["max_wallboxes_without_balancing"], installed)
wb_with_lb = min(simult, installed)

st.markdown(f"""
<div class="mc-wrap">
  <div class="mc" style="animation: fu{_rid} 0.35s ease-out 0.05s both">
    <div class="mc-label">Dostupný výkon pro EV</div>
    <div class="mc-value" style="animation: pi{_rid} 0.45s cubic-bezier(0.34,1.56,0.64,1) 0.10s both">{avail_kw} kW</div>
    <div class="mc-delta">z {total_kw} kW celkem</div>
  </div>
  <div class="mc" style="animation: fu{_rid} 0.35s ease-out 0.12s both">
    <div class="mc-label">Wallboxy bez řízení výkonu</div>
    <div class="mc-value" style="animation: pi{_rid} 0.45s cubic-bezier(0.34,1.56,0.64,1) 0.17s both">{wb_no_lb}</div>
    <div class="mc-delta">nabíjí najednou při plných 11 kW</div>
  </div>
  <div class="mc" style="animation: fu{_rid} 0.35s ease-out 0.19s both">
    <div class="mc-label">Wallboxy s load balancingem</div>
    <div class="mc-value" style="animation: pi{_rid} 0.45s cubic-bezier(0.34,1.56,0.64,1) 0.24s both">{wb_with_lb}</div>
    <div class="mc-delta">nabíjí najednou (sdílí {avail_kw} kW)</div>
  </div>
  <div class="mc" style="animation: fu{_rid} 0.35s ease-out 0.26s both">
    <div class="mc-label">Doporučený počet k instalaci</div>
    <div class="mc-value" style="animation: pi{_rid} 0.45s cubic-bezier(0.34,1.56,0.64,1) 0.31s both;
                                color:{'#198754' if installed > 0 else '#dc3545'}">{installed}</div>
    <div class="mc-delta">pro {expected_evs} EV v domě</div>
  </div>
</div>
""", unsafe_allow_html=True)

if avail_kw == 0:
    st.error(
        f"Při jističi {breaker_amps} A a {flats} bytech nemá budova volný výkon pro EV nabíjení. "
        "Zvažte silnější přípojku nebo snižte počet bytů (pokud zadáváte bytový dům s menší přípojkou). "
        "Alternativně kontaktujte PREdistribuce pro navýšení jisticího příkonu."
    )
elif has_pv and pv_kwp > 0:
    st.info(
        f"FVE {pv_kwp} kWp přidá přes den až {capacity['pv_peak_kw']} kW navíc. "
        "Tato kapacita NENÍ zahrnuta do garantovaného večerního výkonu výše — "
        "slouží pro přesun nabíjení do denních hodin."
    )

st.divider()

# ── Recommendation ────────────────────────────────────────────────────────────
has_mistral = bool(os.environ.get("MISTRAL_API_KEY"))
col_rec, col_risk = st.columns([2, 1])

with col_rec:
    st.subheader("Doporučená konfigurace")
    r = recommendation

    sol_color = "#198754" if r["recommended_solution_type"] != "none_monitor" else "#dc3545"
    lb_icon = {"none": "⛔", "static": "⚙️", "dynamic": "🔄"}.get(r["load_balancing_type"], "")

    # Note beneath the rec table explaining LB power sharing
    lb_note = ""
    if r["number_of_wallboxes"] > 0 and r["load_balancing_type"] != "none":
        lb_note = (
            f"Každý wallbox má nominální výkon 11 kW. "
            f"{r['load_balancing_label']} průběžně sdílí dostupných <strong>{avail_kw} kW</strong> "
            f"mezi aktivními wallboxy — najednou může nabíjet max. <strong>{wb_with_lb} vozů</strong>."
        )
    elif r["number_of_wallboxes"] > 0:
        lb_note = (
            f"Bez řízení výkonu: každý wallbox čerpá 11 kW. "
            f"Zároveň mohou nabíjet max. {wb_no_lb} wallboxy "
            f"(limit přípojky {avail_kw} kW)."
        )

    st.markdown(f"""
<div class="recommendation-box" style="animation: fu{_rid} 0.4s ease-out both">
<h3 style="margin:0 0 14px 0; color:{sol_color}; font-size:1.1rem">
  {r['solution_label']}
</h3>
<table class="rec-table">
  <tr>
    <td class="rec-label">Wallboxy k instalaci</td>
    <td class="rec-val"><span class="rec-val-big"
        style="animation: pi{_rid} 0.5s cubic-bezier(0.34,1.56,0.64,1) 0.15s both">{r['number_of_wallboxes']}</span></td>
  </tr>
  <tr>
    <td class="rec-label">Výkon sítě pro EV</td>
    <td class="rec-val"><span class="rec-val-big"
        style="animation: pi{_rid} 0.5s cubic-bezier(0.34,1.56,0.64,1) 0.22s both">{avail_kw} kW</span>
      <span style="font-size:0.78rem; color:#6c757d; font-weight:400"> dostupných</span></td>
  </tr>
  <tr>
    <td class="rec-label">Simultánně aktivních</td>
    <td class="rec-val">{wb_with_lb} wallboxů najednou</td>
  </tr>
  <tr>
    <td class="rec-label">Řízení výkonu</td>
    <td class="rec-val">{lb_icon} {r['load_balancing_label']}</td>
  </tr>
  <tr>
    <td class="rec-label">Zóna Praha</td>
    <td class="rec-val">{zone_row_adj['grid_zone_id']} &nbsp;
      (rezerva sítě {float(zone_row_adj.get('reserve_margin_pct_2025_synthetic', 0)):.0f} %)</td>
  </tr>
</table>
{f'<p class="rec-note">ℹ️ {lb_note}</p>' if lb_note else ""}
</div>
""", unsafe_allow_html=True)

    # ── AI explanation ─────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    if has_mistral:
        btn_label = (
            "🔄 Přegenerovat AI analýzu"
            if "ai_explanation" in st.session_state
            else "✨ Vysvětlit doporučení pomocí AI"
        )
        if st.button(btn_label, type="primary", use_container_width=True):
            with st.spinner("Mistral AI analyzuje vaše doporučení…"):
                ai_text = generate_explanation(
                    recommendation, capacity, selected_zone, zone_row_adj, user_input, economics
                )
            st.session_state["ai_explanation"] = ai_text
            st.rerun()
        if "ai_explanation" in st.session_state:
            st.markdown(st.session_state["ai_explanation"])
    else:
        with st.expander("Vysvětlení doporučení"):
            st.markdown(r["explanation"])

with col_risk:
    st.subheader("Skóre rizika projektu")
    rs = recommendation["risk_score"]
    color = risk_color(rs)
    risk_label = "Nízké" if rs < 30 else "Střední" if rs < 60 else "Vysoké"
    st.markdown(f"""
<div style="text-align:center; padding:16px 0; animation: fu{_rid} 0.4s ease-out both">
  <div style="font-size:3.5rem; font-weight:800; color:{color}; line-height:1;
              animation: pi{_rid} 0.5s cubic-bezier(0.34,1.56,0.64,1) 0.1s both">{rs}</div>
  <div style="color:#6c757d; font-size:0.85rem; margin-bottom:8px">/ 100 — {risk_label}</div>
  <div class="risk-bar-wrap">
    <div style="width:{rs}%; background:{color}; height:14px; border-radius:8px;
                animation: bf{_rid} 0.7s cubic-bezier(0.25,0.46,0.45,0.94) 0.15s both;
                clip-path: inset(0 0 0 0)"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    reserve_margin = float(zone_row_adj.get("reserve_margin_pct_2025_synthetic", 0))
    grid_sens      = float(zone_row_adj.get("grid_sensitivity_index_derived", 0))
    overload_prob  = float(zone_row_adj.get("target_overload_probability_2030_synthetic", 0))

    st.markdown(f"""
| Ukazatel | Hodnota |
|----------|---------|
| Rezerva sítě | {reserve_margin:.1f} % |
| Citlivost sítě | {grid_sens:.3f} |
| Riziko přetížení 2030 | {overload_prob * 100:.0f} % |
""")

st.divider()

# ── ML debug panel ────────────────────────────────────────────────────────────
with st.expander("🔬 Debug: výstupy AI modelu", expanded=False):
    dbg = ml_prediction.get("debug", {})
    if not dbg:
        st.info("Debug data nejsou dostupná.")
    else:
        mt = dbg.get("model_types", {})
        st.markdown(f"""
**Architektura modelů:**
- **Typ řešení:** `{mt.get('solution', '?')}` (300 stromů, max_depth=5, learning_rate=0.08)
- **Load balancing:** `{mt.get('lb', '?')}` (200 stromů, max_depth=4, learning_rate=0.08)
- **Počet wallboxů:** `{mt.get('wallbox', '?')}` — neuronová síť (vrstvy 64→32→16, ReLU aktivace)

**Expert label zóny:** `{ml_prediction.get('zone_expert_label', '?')}` →
rank {ml_prediction.get('zone_solution_rank', '?')} |
**NN odhad wallboxů:** {ml_prediction.get('wallbox_count_raw', '?')} → zaokrouhleno {ml_prediction.get('wallbox_count_nn', '?')}
""")

        d1, d2 = st.columns(2)

        with d1:
            st.markdown("**Vstupní příznaky (feature vector)**")
            fv = dbg["feature_vector"]
            ZONE_F = {"residential_index_derived", "grid_sensitivity_index_derived",
                      "reserve_capacity_kw_2025_synthetic", "reserve_margin_pct_2025_synthetic",
                      "no_private_parking_index_derived", "target_overload_probability_2030_synthetic",
                      "zone_solution_rank"}
            CAP_F = {"available_for_ev_kw", "total_building_capacity_kw",
                     "max_simultaneous_with_balancing"}
            fv_rows = []
            for k, v in fv.items():
                group = "🏙️ Zóna" if k in ZONE_F else "⚡ Kapacita" if k in CAP_F else "🏠 Budova"
                fv_rows.append({"Skupina": group, "Příznak": k, "Hodnota": round(float(v), 4)})
            st.dataframe(pd.DataFrame(fv_rows), use_container_width=True, hide_index=True)

        with d2:
            st.markdown("**Pravděpodobnosti tříd — typ řešení**")
            sol_df = pd.DataFrame([
                {"Třída": k, "Pravděpodobnost": v,
                 "✓": "←" if k == ml_prediction["solution_type"] else ""}
                for k, v in sorted(dbg["solution_proba"].items(), key=lambda x: -x[1])
            ])
            st.dataframe(sol_df, use_container_width=True, hide_index=True)

            st.markdown("**Pravděpodobnosti tříd — load balancing**")
            lb_df = pd.DataFrame([
                {"Třída": k, "Pravděpodobnost": v,
                 "✓": "←" if k == ml_prediction["load_balancing_type"] else ""}
                for k, v in sorted(dbg["lb_proba"].items(), key=lambda x: -x[1])
            ])
            st.dataframe(lb_df, use_container_width=True, hide_index=True)

        st.markdown("**Důležitost příznaků — klasifikátor řešení**")
        imp_df = pd.DataFrame([
            {"Příznak": k, "Důležitost": v}
            for k, v in sorted(dbg["solution_importances"].items(), key=lambda x: -x[1])
        ])
        st.bar_chart(imp_df.set_index("Příznak")["Důležitost"])

        st.markdown("**Důležitost příznaků — klasifikátor load balancingu**")
        lb_imp_df = pd.DataFrame([
            {"Příznak": k, "Důležitost": v}
            for k, v in sorted(dbg["lb_importances"].items(), key=lambda x: -x[1])
        ])
        st.bar_chart(lb_imp_df.set_index("Příznak")["Důležitost"])

st.divider()

# ── 24h charging plan chart ───────────────────────────────────────────────────
st.subheader("Kdy nabíjet? — 24hodinový plán")
st.caption(
    "Graf ukazuje, kdy doporučujeme v průběhu dne EV nabíjet. "
    "Zelené pásmo = ideální čas (noční tarif, přebytek FVE). "
    "Červené pásmo 18–21 hod = síťová špička — nabíjení v tento čas se nedoporučuje."
)

max_y = max(plan_df["recommended_charging_kw"].max(), pv_kwp * 0.5 if has_pv else 0, avail_kw * 0.3, 5)

fig = go.Figure()

fig.add_vrect(x0=-0.5, x1=5.5, fillcolor="rgba(25,135,84,0.07)", line_width=0,
              annotation_text="Noční okno", annotation_position="top left",
              annotation_font_color="#198754", annotation_font_size=11)
if has_pv:
    fig.add_vrect(x0=9.5, x1=15.5, fillcolor="rgba(255,193,7,0.09)", line_width=0,
                  annotation_text="FVE přebytek", annotation_position="top left",
                  annotation_font_color="#b45309", annotation_font_size=11)
fig.add_vrect(x0=17.5, x1=21.5, fillcolor="rgba(220,53,69,0.10)", line_width=0,
              annotation_text="Špička — vyhýbat se", annotation_position="top right",
              annotation_font_color="#dc3545", annotation_font_size=11)
fig.add_vrect(x0=21.5, x1=23.5, fillcolor="rgba(25,135,84,0.07)", line_width=0)

if has_pv and pv_kwp > 0:
    fig.add_trace(go.Scatter(
        x=plan_df["hour"], y=plan_df["pv_generation_kw"],
        name="FVE výroba (kW)",
        mode="lines",
        line=dict(color="#d97706", width=2, dash="dot"),
        fill="tozeroy",
        fillcolor="rgba(253,186,116,0.18)",
        hovertemplate="%{x}:00 — FVE: %{y:.1f} kW<extra></extra>",
    ))

bar_colors = []
for h in plan_df["hour"]:
    if h in PEAK_HOURS:
        bar_colors.append("#dc3545")
    elif h >= 22 or h <= 5:
        bar_colors.append("#198754")
    elif 10 <= h <= 15 and has_pv:
        bar_colors.append("#d97706")
    else:
        bar_colors.append("#2c7be5")

fig.add_trace(go.Bar(
    x=plan_df["hour"],
    y=plan_df["recommended_charging_kw"],
    name="Doporučené nabíjení (kW)",
    marker_color=bar_colors,
    hovertemplate="%{x}:00 — %{y:.1f} kW<br>%{customdata}<extra></extra>",
    customdata=plan_df["reason"],
))

fig.update_layout(
    xaxis=dict(
        title="Hodina dne",
        tickmode="array",
        tickvals=list(range(0, 24, 2)),
        ticktext=[f"{h}:00" for h in range(0, 24, 2)],
        range=[-0.5, 23.5],
        gridcolor="rgba(0,0,0,0.06)",
    ),
    yaxis=dict(
        title="Výkon (kW)",
        range=[0, max_y * 1.25],
        gridcolor="rgba(0,0,0,0.06)",
    ),
    barmode="overlay",
    legend=dict(orientation="h", y=1.10, x=0),
    margin=dict(t=50, b=50),
    height=400,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#1a202c",
    showlegend=True,
    transition={"duration": 500, "easing": "cubic-in-out"},
)

st.plotly_chart(fig, use_container_width=True, key="charging_plan")

total_charged = plan_df["recommended_charging_kw"].sum()
night_kwh = plan_df[plan_df["hour"].apply(lambda h: h >= 22 or h <= 5)]["recommended_charging_kw"].sum()

col_a, col_b, col_c = st.columns(3)
col_a.metric("Naplánováno celkem", f"{total_charged:.1f} kWh",
             delta=f"potřeba: {expected_daily_kwh:.0f} kWh", delta_color="off")
col_b.metric("Z toho v noci (22–06)", f"{night_kwh:.1f} kWh",
             delta="noční tarif — nejnižší cena", delta_color="off")
if has_pv:
    col_c.metric("Z toho přes FVE (10–15)", f"{pv_kwh_from_plan:.1f} kWh",
                 delta="nulové náklady za energii", delta_color="off")
else:
    col_c.metric("FVE", "Nemáte", delta="aktivujte pro denní nabíjení", delta_color="off")

if total_charged < expected_daily_kwh * 0.8:
    st.warning(
        f"Dostupná kapacita sítě a budovy nestačí pokrýt celou denní potřebu {expected_daily_kwh:.0f} kWh "
        f"({expected_evs} × 12 kWh). Naplánováno: {total_charged:.1f} kWh. "
        "Zvažte silnější přípojku nebo rozložení nabíjení přes více dnů."
    )

st.divider()

# ── Economics ─────────────────────────────────────────────────────────────────
st.subheader("Ekonomika projektu")

ec1, ec2, ec3 = st.columns(3)
with ec1:
    pv_note = ""
    if has_pv and economics["pv_fraction"] > 0:
        pv_pct = economics["pv_fraction"] * 100
        pv_note = f" (vč. {pv_pct:.0f} % zdarma z FVE)"
    st.metric(
        "Měsíční úspora oproti veřejnému nabíjení",
        f"{economics['monthly_savings']:,.0f} Kč",
        delta=f"{expected_evs} aut × 300 kWh{pv_note}",
    )
with ec2:
    st.metric("Celkové náklady instalace (odhad)",
              f"{economics['total_cost']:,.0f} Kč")
    parts = [f"Projekt+instalace: {economics['base_cost']:,.0f} Kč",
             f"Wallboxy: {economics['wallbox_cost']:,.0f} Kč"]
    if economics["lb_cost"]:
        parts.append(f"Load balancing: {economics['lb_cost']:,.0f} Kč")
    if economics["pv_cost"]:
        parts.append(f"Integrace FVE: {economics['pv_cost']:,.0f} Kč")
    st.caption(" • ".join(parts))
with ec3:
    pb = economics["payback_years"]
    pb_str = f"{pb:.1f} let" if pb < 100 else "N/A"
    st.metric("Orientační návratnost", pb_str)
    st.metric("Roční úspora", f"{economics['annual_savings']:,.0f} Kč")

if has_pv and economics["pv_fraction"] > 0:
    st.info(
        f"Z FVE pokryjete přibližně **{economics['pv_fraction']*100:.0f} %** dobíjení "
        f"({economics['monthly_kwh_pv']} kWh/měs. zdarma ze slunce). "
        f"Zbývajících {economics['monthly_kwh_grid']} kWh/měs. z distribuční sítě za 6 Kč/kWh."
    )

st.divider()

# ── Document export ───────────────────────────────────────────────────────────
st.subheader("Podklad pro SVJ")
st.write("Vygenerujte Word dokument s návrhem usnesení, kalkulací a technickými parametry pro hlasování SVJ.")

if st.button("📄 Vygenerovat podklad pro SVJ", type="primary"):
    with st.spinner("Generuji dokument…"):
        doc_buf = generate_svj_document(
            user_input=user_input,
            capacity_result=capacity,
            recommendation=recommendation,
            economics=economics,
            zone_id=selected_zone,
        )
    st.download_button(
        label="⬇️ Stáhnout dokument (.docx)",
        data=doc_buf,
        file_name=f"voltik_svj_{selected_zone}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    st.success("Dokument je připraven ke stažení.")

st.divider()

# ── Chat ──────────────────────────────────────────────────────────────────────
st.subheader("💬 Zeptejte se Voltíka")
if not has_mistral:
    st.info("Chat je dostupný po nastavení proměnné prostředí `MISTRAL_API_KEY`.")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt_text := st.chat_input("Napište dotaz k instalaci wallboxů…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)
        with st.chat_message("assistant"):
            with st.spinner("Přemýšlím…"):
                answer = chat_response(
                    prompt_text, recommendation, capacity, selected_zone, user_input
                )
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

st.divider()
st.caption(
    "Voltík je hackathonový demonstrátor. Data o distribuční síti pocházejí ze syntetického datasetu "
    "pražských zón. Doporučení jsou orientační odhady – nejsou náhradou za elektroprojekt ani "
    "souhlas distributora. | © 2025 Voltík Team"
)
