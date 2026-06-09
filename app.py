import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voltík – EV poradce pro SVJ",
    page_icon="⚡",
    layout="wide",
)

# ── CSS tweaks ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 8px;
    border-left: 4px solid #4c9be8;
}
.metric-card h3 { margin: 0 0 4px 0; font-size: 0.85rem; color: #aaa; }
.metric-card p  { margin: 0; font-size: 1.8rem; font-weight: 700; color: #fff; }
.risk-bar-bg { background: #333; border-radius: 8px; height: 12px; width: 100%; }
.recommendation-box {
    background: #162032;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #2a4a6a;
    margin-top: 12px;
}
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
    grid_cap_df = load_grid_capacity()
    solutions_df = load_candidate_solutions()
    classifier = load_or_train_classifier()

zone_ids = sorted(zones_df["grid_zone_id"].unique())

# ── Sidebar form ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parametry domu")

    flats = st.slider("Počet bytových jednotek", 4, 200, 24)
    parking_spaces = st.slider("Počet parkovacích míst", 0, 200, 12)
    breaker_amps = st.selectbox(
        "Hlavní jistič (A)",
        [25, 32, 40, 50, 63, 80, 100, 125, 160],
        index=4,
        format_func=lambda x: f"3×{x} A",
    )
    expected_evs = st.slider("Očekávaný počet EV v domě", 0, 100, 5)

    st.divider()
    has_pv = st.checkbox("Střešní fotovoltaika (FVE)", value=False)
    pv_kwp = 0.0
    if has_pv:
        pv_kwp = st.number_input("Výkon FVE (kWp)", min_value=0.0, max_value=500.0, value=20.0, step=1.0)

    st.divider()
    selected_zone = st.selectbox("Distribuční zóna Praha", zone_ids, index=0)
    scenario = st.selectbox(
        "Scénář adopce EV do 2030",
        ["conservative", "central", "ambitious"],
        index=1,
        format_func=lambda s: {"conservative": "Konzervativní", "central": "Střední", "ambitious": "Ambiciózní"}[s],
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

capacity = calculate_wallbox_capacity(
    flats, parking_spaces, breaker_amps, expected_evs, has_pv, pv_kwp
)

zone_row = get_zone_row(zones_df, selected_zone)
ml_solution = predict_solution(classifier, zone_row)
recommendation = recommend_configuration(user_input, zone_row, capacity, ml_solution)
economics = calculate_economics(
    expected_evs,
    recommendation["number_of_wallboxes"],
    has_pv,
    recommendation["load_balancing_type"],
)

expected_daily_kwh = expected_evs * 12  # ~12 kWh/day per EV average
plan_df = create_hourly_charging_plan(
    selected_zone,
    expected_daily_kwh,
    has_pv,
    pv_kwp,
    hourly_agg,
    capacity["available_for_ev_kw"],
)

# ── Four metric cards ─────────────────────────────────────────────────────────
st.subheader("Kapacitní přehled")
c1, c2, c3, c4 = st.columns(4)

wl = capacity["warning_level"]
with c1:
    st.metric(
        "Dostupný výkon pro EV",
        f"{capacity['available_for_ev_kw']} kW",
        delta=f"z celkových {capacity['total_building_capacity_kw']} kW",
    )
with c2:
    st.metric(
        "Wallboxy bez load balancingu",
        capacity["max_wallboxes_without_balancing"],
        delta="při 11 kW / wallbox",
    )
with c3:
    st.metric(
        "Wallboxy s dyn. load balancingem",
        capacity["max_simultaneous_with_balancing"],
        delta="současně aktivních",
    )
with c4:
    level_text = {"green": "Nízké", "orange": "Střední", "red": "Vysoké"}[wl]
    st.metric(
        f"Riziko přetížení {warning_emoji(wl)}",
        level_text,
        delta=f"zóna {selected_zone}",
    )

if has_pv and pv_kwp > 0:
    st.info(
        f"FVE {pv_kwp} kWp přidá až {capacity['pv_peak_kw']} kW přes den. "
        "Tento výkon není zahrnut do garantované večerní kapacity."
    )

st.divider()

# ── Recommendation card ───────────────────────────────────────────────────────
col_rec, col_risk = st.columns([2, 1])

with col_rec:
    st.subheader("Doporučená konfigurace")
    r = recommendation
    st.markdown(f"""
<div class="recommendation-box">
<h3 style="margin:0 0 12px 0; color:#4c9be8">
  {r['solution_label']}
</h3>
<table style="width:100%; border-collapse:collapse">
  <tr><td style="color:#aaa; padding:4px 0">Počet wallboxů</td>
      <td style="font-weight:700; text-align:right">{r['number_of_wallboxes']}</td></tr>
  <tr><td style="color:#aaa; padding:4px 0">Celkový výkon</td>
      <td style="font-weight:700; text-align:right">{r['recommended_total_kw']:.0f} kW</td></tr>
  <tr><td style="color:#aaa; padding:4px 0">Load balancing</td>
      <td style="font-weight:700; text-align:right">{r['load_balancing_label']}</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    with st.expander("Vysvětlení doporučení"):
        st.markdown(r["explanation"])

with col_risk:
    st.subheader("Skóre rizika")
    rs = recommendation["risk_score"]
    color = risk_color(rs)
    st.markdown(f"""
<div style="text-align:center; padding: 20px 0">
  <div style="font-size:3rem; font-weight:700; color:{color}">{rs}</div>
  <div style="color:#aaa; font-size:0.9rem">/ 100</div>
  <div style="margin-top:8px; background:#333; border-radius:8px; height:12px">
    <div style="width:{rs}%; background:{color}; height:12px; border-radius:8px"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    reserve_margin = float(zone_row.get("reserve_margin_pct_2025_synthetic", 0))
    grid_sens = float(zone_row.get("grid_sensitivity_index_derived", 0))
    st.markdown(f"**Rezerva sítě:** {reserve_margin:.1f} %")
    st.markdown(f"**Citlivost sítě:** {grid_sens:.2f}")

st.divider()

# ── Hourly charging chart ─────────────────────────────────────────────────────
st.subheader("Doporučený 24h plán nabíjení")

peak_shape = [
    {"type": "rect", "x0": 17.5, "x1": 21.5,
     "y0": 0, "y1": plan_df["recommended_charging_kw"].max() * 1.2 or 10,
     "fillcolor": "rgba(231,76,60,0.12)", "line": {"width": 0}, "layer": "below"}
]

fig = go.Figure()

if has_pv:
    fig.add_trace(go.Bar(
        x=plan_df["hour"], y=plan_df["pv_generation_kw"],
        name="FVE výroba (kW)",
        marker_color="rgba(241,196,15,0.5)",
        hovertemplate="Hod. %{x}:00<br>FVE: %{y:.1f} kW<extra></extra>",
    ))

fig.add_trace(go.Bar(
    x=plan_df["hour"], y=plan_df["recommended_charging_kw"],
    name="Doporučené nabíjení (kW)",
    marker_color="#4c9be8",
    hovertemplate="Hod. %{x}:00<br>Nabíjení: %{y:.1f} kW<br>%{customdata}<extra></extra>",
    customdata=plan_df["reason"],
))

fig.add_vrect(
    x0=17.5, x1=21.5,
    fillcolor="rgba(231,76,60,0.10)", line_width=0,
    annotation_text="Síťová špička – vyhýbat se",
    annotation_position="top left",
)

fig.update_layout(
    xaxis=dict(title="Hodina", tickmode="linear", tick0=0, dtick=1),
    yaxis=dict(title="kW"),
    barmode="overlay",
    legend=dict(orientation="h", y=1.08),
    margin=dict(t=40, b=40),
    height=380,
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="#fff",
)

st.plotly_chart(fig, use_container_width=True)

total_charged = plan_df["recommended_charging_kw"].sum()
st.caption(
    f"Plánované denní nabíjení: **{total_charged:.1f} kWh** "
    f"(cílová potřeba: {expected_daily_kwh:.0f} kWh). "
    "Nabíjení je soustředěno do hodin s nejlepší dostupností sítě."
)

st.divider()

# ── Economics ─────────────────────────────────────────────────────────────────
st.subheader("Ekonomické srovnání")

ec1, ec2, ec3 = st.columns(3)
with ec1:
    st.metric("Měsíční úspora vs. veřejné nabíjení", f"{economics['monthly_savings']:,.0f} Kč")
    st.caption(f"({expected_evs} vozidel × 250 kWh × 6 Kč/kWh rozdíl)")
with ec2:
    st.metric("Celkové náklady na instalaci (odhad)", f"{economics['total_cost']:,.0f} Kč")
    cost_items = []
    cost_items.append(f"Projekt a instalace: {economics['base_cost']:,.0f} Kč")
    cost_items.append(f"Wallboxy: {economics['wallbox_cost']:,.0f} Kč")
    if economics["lb_cost"]:
        cost_items.append(f"Load balancing: {economics['lb_cost']:,.0f} Kč")
    if economics["pv_cost"]:
        cost_items.append(f"Integrace FVE: {economics['pv_cost']:,.0f} Kč")
    st.caption(" • ".join(cost_items))
with ec3:
    pb = economics["payback_years"]
    pb_str = f"{pb:.1f} let" if pb < 100 else "N/A"
    st.metric("Orientační návratnost", pb_str)
    st.metric("Roční úspora", f"{economics['annual_savings']:,.0f} Kč")

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
st.caption(
    "Voltík je hackathonový demonstrátor. Data o distribuční síti pocházejí ze syntetického datasetu "
    "pražských zón. Doporučení jsou orientační odhady – nejsou náhradou za elektroprojekt ani "
    "souhlas distributora. | © 2025 Voltík Team"
)
