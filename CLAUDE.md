# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The ML classifier (`models/classifier.joblib`) is trained automatically on first run if missing (~10 s). To retrain manually:

```bash
python src/train_models.py
```

There is no test suite and no linter configured.

## Architecture

`app.py` is the single Streamlit entrypoint. It calls the `src/` modules in order and renders everything in one pass — there is no routing, session state persistence beyond widget values, or background threads.

**Data flow through the app on each interaction:**

1. `data_loader.py` — loads and caches all CSVs via `@st.cache_data`. The hourly file (2.3 M rows) is aggregated to ~82 k zone×hour rows at load time and never kept in full.
2. `capacity_calculator.py` — pure math, no data dependency. Takes user form values, returns a dict with kW figures and a `warning_level` (`green`/`orange`/`red`).
3. `train_models.py` — loads or trains `RandomForestClassifier` on `zones_train.csv` (12 zone-level features → `target_recommended_solution_synthetic`). Validated on `zones_validation.csv`. Saved with `joblib`.
4. `recommender.py` — takes the ML prediction and applies hardcoded safety overrides (e.g. `available_for_ev_kw < 7` → force `none_monitor`; `grid_sensitivity_index_derived > 0.75` → force dynamic load balancing). Returns Czech explanation strings.
5. `scheduler.py` — builds a 24-row DataFrame (one per hour) using a scoring function: `score = avg_available_kw − 200 × overload_rate + pv_bonus − 100 for peak hours 18–21`. Allocates `expected_daily_kwh` greedily across highest-scored hours.
6. `document_generator.py` — generates an in-memory `.docx` (never written to disk unless explicitly saved) returned as `BytesIO` for Streamlit's `download_button`.
7. `utils.py` — economics calculation (savings vs. public charging, payback period) and colour helpers for the UI.

## Key data facts

- `hourly_grid_and_charging_history_2025.csv` — 2.3 M rows; always read with `usecols` to limit memory. Never load the full file in a new code path.
- `zones_train.csv` / `zones_validation.csv` — concatenated into a single DataFrame in `data_loader.load_zones()`. The `split` column distinguishes them but is not used after loading.
- All capacity/load values in CSVs are `_synthetic` (modelled); only columns tagged `_real` come from actual measurements.
- The classifier is trained only on zone-level features, not on building parameters. Building-level math is purely in `capacity_calculator.py`.

## Positioning constraint

Voltík must not claim to have real building electrical measurements. It is a *pre-assessment* tool. Any new recommendation text should include a qualifier that output requires validation by a certified electrician and PREdistribuce.
