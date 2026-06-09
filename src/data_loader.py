import pandas as pd
import streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

HOURLY_COLS = [
    "grid_zone_id", "hour", "representative_season", "temperature_c_real",
    "base_load_kw_synthetic", "ev_charging_load_kw_synthetic",
    "available_capacity_after_ev_kw_synthetic", "overload_flag_synthetic",
]


@st.cache_data(show_spinner="Načítám data zón…")
def load_zones():
    train = pd.read_csv(DATA_DIR / "zones_train.csv")
    val = pd.read_csv(DATA_DIR / "zones_validation.csv")
    return pd.concat([train, val], ignore_index=True)


@st.cache_data(show_spinner="Načítám hodinová data…")
def load_hourly_aggregated():
    """Load hourly data and aggregate by zone+hour to avoid keeping 2M rows in memory."""
    df = pd.read_csv(
        DATA_DIR / "hourly_grid_and_charging_history_2025.csv",
        usecols=HOURLY_COLS,
    )
    agg = (
        df.groupby(["grid_zone_id", "hour"])
        .agg(
            avg_available_kw=("available_capacity_after_ev_kw_synthetic", "mean"),
            avg_base_load_kw=("base_load_kw_synthetic", "mean"),
            avg_ev_load_kw=("ev_charging_load_kw_synthetic", "mean"),
            overload_rate=("overload_flag_synthetic", "mean"),
        )
        .reset_index()
    )
    return agg


@st.cache_data(show_spinner=False)
def load_grid_capacity():
    return pd.read_csv(DATA_DIR / "grid_capacity_and_reserve_2025.csv")


@st.cache_data(show_spinner=False)
def load_candidate_solutions():
    return pd.read_csv(DATA_DIR / "candidate_solutions.csv")


@st.cache_data(show_spinner=False)
def load_future_scenarios():
    return pd.read_csv(DATA_DIR / "future_scenarios.csv")


def get_zone_row(zones_df: pd.DataFrame, grid_zone_id: str) -> pd.Series:
    row = zones_df[zones_df["grid_zone_id"] == grid_zone_id]
    if row.empty:
        raise ValueError(f"Zone {grid_zone_id} not found")
    return row.iloc[0]
