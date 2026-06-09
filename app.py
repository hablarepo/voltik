import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os
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

st.markdown("""
<style>
.recommendation-box {
    background: #162032;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #2a4a6a;
    margin-top: 12px;
}
.capacity-ok  { color: #27ae60; font-weight: 700; }
.capacity-warn { color: #f39c12; font-weight: 700; }
.capacity-bad  { color: #e74c3c; font-weight: 700; }
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
BREAKER_OPTIONS = [25, 32, 40, 50, 63, 80, 100, 125, 160, 200, 250]

with st.sidebar:
    st.header("Parametry domu")

    flats = st.slider("Počet bytových jednotek", 4, 200, 24)
    parking_spaces = st.slider("Počet parkovacích míst", 0, 200, 12)
    breaker_amps = st.selectbox(
        "Hlavní jistič budovy",
        BREAKER_OPTIONS,
        index=6,                          # default 100 A
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
    selected_zone = st.selectbox("Distribuční zóna Praha", zone_ids, index=0)
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

capacity = calculate_wallbox_capacity(
    flats, parking_spaces, breaker_amps, expected_evs, has_pv, pv_kwp
)

zone_row = get_zone_row(zones_df, selected_zone)
ml_prediction = predict_solution(classifier, zone_row, capacity, user_input)
recommendation = recommend_configuration(user_input, zone_row, capacity, ml_prediction)
economics = calculate_economics(
    expected_evs,
    recommendation["number_of_wallboxes"],
    has_pv,
    recommendation["load_balancing_type"],
)

expected_daily_kwh = expected_evs * 12
plan_df = create_hourly_charging_plan(
    selected_zone,
    expected_daily_kwh,
    has_pv,
    pv_kwp,
    hourly_agg,
    capacity["available_for_ev_kw"],
)

# ── Capacity section ──────────────────────────────────────────────────────────
st.subheader("Kapacitní přehled")

wl = capacity["warning_level"]
wl_color = {"green": "#27ae60", "orange": "#f39c12", "red": "#e74c3c"}[wl]
wl_text = {"green": "Kapacita dostačující", "orange": "Nutný load balancing", "red": "Kapacita nedostačující"}[wl]

# Capacity bar showing base load vs available headroom
total_kw = capacity["total_building_capacity_kw"]
base_kw = capacity["estimated_evening_base_load_kw"]
avail_kw = capacity["available_for_ev_kw"]
base_pct = min(100, base_kw / total_kw * 100) if total_kw > 0 else 100
avail_pct = min(100 - base_pct, avail_kw / total_kw * 100) if total_kw > 0 else 0

st.markdown(
    f"**Přípojka budovy: {total_kw} kW** &nbsp;|&nbsp; "
    f"Odhadovaná večerní spotřeba: **{base_kw} kW** &nbsp;|&nbsp; "
    f"Volné pro EV: <span style='color:{wl_color}; font-weight:700'>{avail_kw} kW</span> "
    f"— {wl_text} {warning_emoji(wl)}",
    unsafe_allow_html=True,
)

# Visual capacity bar
st.markdown(f"""
<div style="height:20px; border-radius:8px; background:#333; width:100%; margin:6px 0 16px 0; position:relative">
  <div style="position:absolute; left:0; width:{base_pct:.1f}%; height:100%;
       background:#c0392b; border-radius:8px 0 0 8px"></div>
  <div style="position:absolute; left:{base_pct:.1f}%; width:{avail_pct:.1f}%; height:100%;
       background:{wl_color}; border-radius:0"></div>
</div>
<div style="display:flex; gap:24px; font-size:0.82rem; color:#aaa; margin-bottom:8px">
  <span><span style="color:#c0392b">■</span> Spotřeba domu ({base_kw} kW)</span>
  <span><span style="color:{wl_color}">■</span> Dostupné pro EV ({avail_kw} kW)</span>
  <span><span style="color:#555">■</span> Rezerva jističe</span>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Dostupný výkon pro EV", f"{avail_kw} kW",
              delta=f"z {total_kw} kW celkem", delta_color="off")
with c2:
    v = capacity["max_wallboxes_without_balancing"]
    st.metric("Wallboxy bez řízení výkonu", v,
              delta="při 11 kW každý", delta_color="off")
with c3:
    v2 = capacity["max_simultaneous_with_balancing"]
    st.metric("Wallboxy s load balancingem", v2,
              delta="současně aktivních při 3,7 kW min.", delta_color="off")
with c4:
    installed = recommendation["number_of_wallboxes"]
    st.metric(
        "Doporučený počet k instalaci",
        installed,
        delta=f"pro {expected_evs} EV v domě",
        delta_color="off",
    )

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

    sol_color = "#27ae60" if r["recommended_solution_type"] != "none_monitor" else "#e74c3c"
    lb_icon = {"none": "⛔", "static": "⚙️", "dynamic": "🔄"}.get(r["load_balancing_type"], "")

    st.markdown(f"""
<div class="recommendation-box">
<h3 style="margin:0 0 14px 0; color:{sol_color}; font-size:1.1rem">
  {r['solution_label']}
</h3>
<table style="width:100%; border-collapse:collapse; font-size:0.97rem">
  <tr style="border-bottom:1px solid #2a4a6a">
    <td style="color:#aaa; padding:6px 0">Wallboxy k instalaci</td>
    <td style="font-weight:700; text-align:right; font-size:1.3rem">{r['number_of_wallboxes']}</td>
  </tr>
  <tr style="border-bottom:1px solid #2a4a6a">
    <td style="color:#aaa; padding:6px 0">Instalovaný výkon</td>
    <td style="font-weight:700; text-align:right">{r['recommended_total_kw']:.0f} kW</td>
  </tr>
  <tr style="border-bottom:1px solid #2a4a6a">
    <td style="color:#aaa; padding:6px 0">Řízení výkonu</td>
    <td style="font-weight:700; text-align:right">{lb_icon} {r['load_balancing_label']}</td>
  </tr>
  <tr>
    <td style="color:#aaa; padding:6px 0">Zóna Praha</td>
    <td style="font-weight:700; text-align:right">{zone_row['grid_zone_id']} &nbsp;
      (rezerva sítě {float(zone_row.get('reserve_margin_pct_2025_synthetic', 0)):.0f} %)</td>
  </tr>
</table>
</div>
""", unsafe_allow_html=True)

    if has_mistral:
        with st.expander("🤖 AI vysvětlení (Mistral)", expanded=True):
            with st.spinner("Generuji vysvětlení…"):
                ai_text = generate_explanation(
                    recommendation, capacity, selected_zone, zone_row, user_input, economics
                )
            st.markdown(ai_text)
    else:
        with st.expander("Vysvětlení doporučení"):
            st.markdown(r["explanation"])

with col_risk:
    st.subheader("Skóre rizika projektu")
    rs = recommendation["risk_score"]
    color = risk_color(rs)
    risk_label = "Nízké" if rs < 30 else "Střední" if rs < 60 else "Vysoké"
    st.markdown(f"""
<div style="text-align:center; padding:16px 0">
  <div style="font-size:3.5rem; font-weight:800; color:{color}; line-height:1">{rs}</div>
  <div style="color:#aaa; font-size:0.85rem; margin-bottom:8px">/ 100 — {risk_label}</div>
  <div style="background:#333; border-radius:8px; height:14px; margin:0 20px">
    <div style="width:{rs}%; background:{color}; height:14px; border-radius:8px"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    reserve_margin = float(zone_row.get("reserve_margin_pct_2025_synthetic", 0))
    grid_sens = float(zone_row.get("grid_sensitivity_index_derived", 0))
    overload_prob = float(zone_row.get("target_overload_probability_2030_synthetic", 0))

    st.markdown(f"""
| Ukazatel | Hodnota |
|----------|---------|
| Rezerva sítě | {reserve_margin:.1f} % |
| Citlivost sítě | {grid_sens:.3f} |
| Riziko přetížení 2030 | {overload_prob * 100:.0f} % |
""")

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

# Background zones
fig.add_vrect(x0=-0.5, x1=5.5, fillcolor="rgba(39,174,96,0.08)", line_width=0,
              annotation_text="Noční okno", annotation_position="top left",
              annotation_font_color="#27ae60", annotation_font_size=11)
if has_pv:
    fig.add_vrect(x0=9.5, x1=15.5, fillcolor="rgba(241,196,15,0.08)", line_width=0,
                  annotation_text="FVE přebytek", annotation_position="top left",
                  annotation_font_color="#f1c40f", annotation_font_size=11)
fig.add_vrect(x0=17.5, x1=21.5, fillcolor="rgba(231,76,60,0.15)", line_width=0,
              annotation_text="Špička — vyhýbat se", annotation_position="top right",
              annotation_font_color="#e74c3c", annotation_font_size=11)
fig.add_vrect(x0=21.5, x1=23.5, fillcolor="rgba(39,174,96,0.08)", line_width=0)

# PV generation line
if has_pv and pv_kwp > 0:
    fig.add_trace(go.Scatter(
        x=plan_df["hour"], y=plan_df["pv_generation_kw"],
        name="FVE výroba (kW)",
        mode="lines",
        line=dict(color="#f1c40f", width=2, dash="dot"),
        fill="tozeroy",
        fillcolor="rgba(241,196,15,0.10)",
        hovertemplate="%{x}:00 — FVE: %{y:.1f} kW<extra></extra>",
    ))

# Recommended charging bars — coloured by time window
bar_colors = []
for h in plan_df["hour"]:
    if h in PEAK_HOURS:
        bar_colors.append("#e74c3c")
    elif (h >= 22 or h <= 5):
        bar_colors.append("#27ae60")
    elif 10 <= h <= 15 and has_pv:
        bar_colors.append("#f1c40f")
    else:
        bar_colors.append("#4c9be8")

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
    ),
    yaxis=dict(title="Výkon (kW)", range=[0, max_y * 1.25]),
    barmode="overlay",
    legend=dict(orientation="h", y=1.10, x=0),
    margin=dict(t=50, b=50),
    height=400,
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="#fff",
    showlegend=True,
)

st.plotly_chart(fig, use_container_width=True)

total_charged = plan_df["recommended_charging_kw"].sum()
night_kwh = plan_df[plan_df["hour"].apply(lambda h: h >= 22 or h <= 5)]["recommended_charging_kw"].sum()
pv_kwh = plan_df[(plan_df["hour"] >= 10) & (plan_df["hour"] <= 15)]["recommended_charging_kw"].sum() if has_pv else 0

col_a, col_b, col_c = st.columns(3)
col_a.metric("Naplánováno celkem", f"{total_charged:.1f} kWh",
             delta=f"potřeba: {expected_daily_kwh:.0f} kWh", delta_color="off")
col_b.metric("Z toho v noci (22–06)", f"{night_kwh:.1f} kWh",
             delta="noční tarif — nejnižší cena", delta_color="off")
if has_pv:
    col_c.metric("Z toho přes FVE (10–15)", f"{pv_kwh:.1f} kWh",
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
    st.metric("Měsíční úspora oproti veřejnému nabíjení",
              f"{economics['monthly_savings']:,.0f} Kč",
              delta=f"{expected_evs} aut × 250 kWh × 6 Kč/kWh rozdíl")
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
