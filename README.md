# Voltík — AI poradce pro nabíjení v bytovém domě

Voltík is a hackathon demo tool for Czech apartment building associations (SVJ).  
It combines user-entered building parameters with Prague grid-zone data to produce a fast feasibility estimate for EV wallbox installation.

> **Voltík is not a replacement for an electrical engineer.**  
> It is a fast AI pre-assessment tool that tells an SVJ whether the project is likely feasible,
> what configuration to discuss, and how to avoid peak-time overload.

## What it does

1. Takes building parameters (flats, breaker size, parking, EV count, optional PV)
2. Looks up the selected Prague grid zone from real distribution data
3. Calculates available EV charging capacity
4. Uses a RandomForest classifier trained on zone data to recommend a charger configuration
5. Builds a 24-hour charging schedule that avoids grid peak hours (18–21)
6. Shows a financial comparison vs. public charging
7. Generates a Word document (SVJ voting template) with the full recommendation

## Data sources

| File | Usage |
|------|-------|
| `zones_train.csv` / `zones_validation.csv` | ML classifier training and Prague grid-zone profiles |
| `hourly_grid_and_charging_history_2025.csv` | 24h grid load profile per zone (aggregated at load time) |
| `grid_capacity_and_reserve_2025.csv` | Grid capacity and reserve margins |
| `candidate_solutions.csv` | Charger configuration reference table |
| `future_scenarios.csv` | EV adoption scenario parameters |

Building data is user-entered. Grid context comes from the provided synthetic zone datasets.  
Recommendations are estimates — not engineering approval.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run will train the ML classifier automatically (~10 seconds) and save it to `models/`.

## Project structure

```
voltik/
  app.py                      Streamlit UI
  requirements.txt
  data/                       CSV files
  src/
    data_loader.py            Cached CSV loading + hourly aggregation
    capacity_calculator.py    Building capacity math
    train_models.py           ML model training / loading
    recommender.py            Rule + ML configuration recommendation
    scheduler.py              24h charging plan builder
    document_generator.py     .docx SVJ template generator
    utils.py                  Economics + colour helpers
  models/                     Saved classifier (auto-generated)
  outputs/                    Generated documents
```

## Disclaimer

This is a hackathon demonstrator. All grid data is synthetic. Recommendations are illustrative and must be validated by a certified electrical engineer and the local electricity distributor (PREdistribuce) before any real installation.
