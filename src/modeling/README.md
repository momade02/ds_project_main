# Modeling - Component README

Table of Contents  
1. [Purpose & quick summary](#1-purpose--quick-summary)  
2. [Where this fits in the pipeline](#2-where-this-fits-in-the-pipeline)  
3. [Inputs & Outputs](#3-inputs--outputs)  
4. [How it works (high level)](#4-how-it-works-high-level)  
5. [Automation hooks](#5-automation-hooks)  
6. [Validation & quality checks](#6-validation--quality-checks)  
7. [Error handling & troubleshooting](#7-error-handling--troubleshooting)
8. [Model selection rationale & interpretability](#8-model-selection-rationale--interpretability) 
9. [Links](#9-links)  

## 1) Purpose & quick summary

`modeling` - Responsible for horizon-aware predictions of fuel prices at gas stations using fitted ARDL (Autoregressive Distributed Lag) models. The component transforms station data with price lags and ETAs into actionable price forecasts for downstream analysis and dashboard presentation.

For non-technical readers: This component uses econometrics to predict fuel prices at stations along a route, so the app can recommend optimal refueling decisions.

## 2) Where this fits in the pipeline

- **Upstream:**  
  - `src/integration` (enriched station data with price lags and ETAs)  
  - `src/data_pipeline` (data sources)
- **Downstream:**  
  - `src/decision` (stop optimization and recommendation logic)  
  - `src/app` (dashboard/UI presentation)

## 3) Inputs & Outputs

- **Inputs**
  - List of station dictionaries, each containing:
    - Required lagged price features (e.g., `price_lag_1d_e5`, `price_lag_2d_e10`, `price_lag_7d_diesel`, etc.)
    - Current price (if available)
    - ETA (estimated time of arrival)
    - Time cell (integer, 0–47 indicating position in current/next day)
    - Other metadata as needed

- **Outputs**
  - Annotated station dictionaries with:
    - Predicted price for each fuel (`pred_price_e5`, `pred_price_e10`, `pred_price_diesel`)
    - Error flags if prediction could not be performed

Example input station dict:

```python
{
  ...existing fields...,
  "price_lag_1d_e5": 1.789,
  "price_lag_2d_e5": 1.799,
  "price_lag_3d_e5": 1.809,
  "price_lag_7d_e5": 1.759,
  "price_current_e5": 1.799,
  "eta": "2026-01-22T15:30:00Z",
  "time_cell": 31,
  "fuel_type":"e5"
  ...
}
```

## 4) How it works (high level)

**Code documentation:** All code throughout this component is extensively commented with inline documentation explaining function logic, variables and rules. For low-level implementation details refer directly to the source code files. Each file contains detailed comments.

### Station-level prediction pipeline

For each station and fuel type, the component executes a consistent pipeline:

#### **Step 1: Spot vs Forecast decision**
- If **ETA ≤ 10 minutes** and realtime price exists → **Spot mode**
  - Predicted price = current realtime price (Tankerkönig)
  - No forecasting model is called
- If **ETA > 10 minutes** → **Forecast mode**
  - System calls a trained ARDL model
  - Selects appropriate horizon based on time distance to arrival

#### **Step 2: Horizon selection (Forecast mode only)**
- Maps ETA to discrete time buckets:
  - A day is split into **48 time cells** (each = 30 minutes)
  - Derives: **Now cell**, **ETA cell**, **Cells ahead** = (ETA cell - Now cell)
  - Accounts for multi-day lookaheads (day wrap)


- **Horizon mapping rule:**
  - **Minutes ahead ≤ 10** → Spot mode (no model)
  - **Minutes ahead 10–40** (1–4 cells ahead) → use h1, h2, h3, or h4 (horizon-specific intraday models)
  - **Minutes ahead > 120** (>4 cells ahead) → use h0_daily (daily-lags-only model)

#### **Step 3: Feature vector construction**
- Gathers required lag features for the selected horizon:
  - **Daily lags** (all horizons): `price_lag_1d`, `price_lag_2d`, `price_lag_3d`, `price_lag_7d` at the same time-of-day cell
  - **Intraday anchor** (h1–h4 only): horizon-specific intraday feature (e.g., `price_lag_1cell`, `price_lag_2cell`, `price_lag_3cell`, `price_lag_4cell`)
  - **Fallback**: If current prices are unavailable, only historical lags are used (no intraday anchor)

#### **Step 4: Model selection and loading**
- Selects the appropriate pre-trained ARDL model from `src/modeling/models/`:
  - 3 fuel types (e5, e10, diesel)
  - 5 horizons (h0_daily, h1_1cell, h2_2cell, h3_3cell, h4_4cell)
  - **Total: 15 models** (each `.joblib` file)
- Models are loaded on-demand and cached (LRU cache) for efficiency

#### **Step 5: Prediction and output**
- Runs the selected ARDL model with the constructed feature vector
- Returns predicted price at arrival (€/L)
- Writes result back into station dict as `pred_price_<fuel>`

## 5) Automation hooks

- Intended triggers:
  - Called directly via Python from the main pipeline script (`predict_all_fuels` function)

- Automation features:
  - Models are loaded automatically and cached (LRU cache) for efficiency
  - All logic is encapsulated; users do not need to modify code

## 6) Validation & quality checks

- Checks for presence of all required lag features before prediction
- Validates fuel type and ETA format
- Verifies model file availability before loading
- Handles timezone conversions correctly (UTC → local)
- Detects and reports incomplete feature vectors

## 7) Error handling & troubleshooting

**Common failure modes:**
- Missing model files: Raises `PredictionError` with remediation instructions
- Invalid fuel type: Raises `ValueError` with valid options
- Incomplete feature vector: Skips prediction, output remains `None`
- ETA parsing errors: Falls back to cell-based logic
- Missing lag inputs: Uses only available features; still produces prediction if core set exists

## 8) Model selection rationale

Before estimating the ARDL models, we experimented with a Gradient Boosting approach (LightGBM) to assess whether a flexible machine-learning model could capture additional predictive structure. The analysis of feature importance (gain) revealed that model performance is largely driven by simple lagged price information, while most other features contribute only marginally.

![Feature importance](../../structure_graphs/Feature_Importance.jpg)

This finding suggests that short-term price dynamics are primarily governed by temporal dependence rather than complex nonlinear interactions. Consequently, we transitioned to a dedicated time-series framework using ARDL models. As a transparent benchmark, we therefore use the previous day’s price as the baseline reference against which more advanced models are evaluated. 

Training & validation data:
- Models trained on historical price data from 01.2023-06.2024
- Validation period: 07.2024-12.2024

| Month   | BaseMAE | e10_h0_daily_MAE | e10_h1_1cell_MAE | e10_h2_2cell_MAE | e10_h3_3cell_MAE | e10_h4_4cell_MAE |
|---------|---------|------------------|------------------|------------------|------------------|------------------|
| 07 | 0.0124  | 0.0122           | 0.0101           | 0.0116           | 0.0118           | 0.0118           |
| 08 | 0.0145  | 0.0149           | 0.0111           | 0.0131           | 0.0135           | 0.0136           |
| 09 | 0.0148  | 0.0159           | 0.0114           | 0.0135           | 0.0141           | 0.0143           |
| 10 | 0.0150  | 0.0149           | 0.0115           | 0.0138           | 0.0142           | 0.0143           |
| 11 | 0.0137  | 0.0135           | 0.0108           | 0.0128           | 0.0131           | 0.0132           |
| 12 | 0.0140  | 0.0133           | 0.0109           | 0.0130           | 0.0134           | 0.0135           |


The ARDL models, especially the h1-h4 models, achieve lower monthly MAE than the simple lag-1 baseline (“yesterday’s price”), indicating a performance gain from explicitly modeling time-series dynamics.
  
## 9) Links

- [Back to Root README](../../README.md)
- Related components:
  - Integration: [../integration/README.md](../integration/README.md)
  - Decision: [../decision/README.md](../decision/README.md)
  - App/UI: [../app/README.md](../app/README.md)