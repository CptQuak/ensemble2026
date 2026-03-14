# Time Series Forecasting Pipeline

## Current Modeling Approach

This project implements a time series forecasting pipeline to predict monthly mean values for individual devices from May 2025 to October 2025, using historical sensor data up to October 2024.

The approach is divided into three main stages: Data Processing, Model Training, and Forecasting.

### 1. Data Processing
- **Engine**: The data manipulation pipeline leverages `polars` for fast lazy evaluation and memory efficiency.
- **Spatial Clustering**: Device locations (`latitude` and `longitude`) are clustered into up to 16 geographic regions using `KMeans` to provide spatial context. 
- **Temporal Aggregation**: Historical data is truncated to the start of the hour. We aggregate target variable `x2` to form `x2_mean`. Noisy variables (`period`, `x1`, `x3`) are dropped.
- **Imputation**: Missing values within each device's temporal sequence are handled via forward and backward filling.
- **Regional Context**: We compute and append region-level hourly averages to capture broader geographic trends for dynamic features.

### 2. Model Training
- **Framework**: `MLForecast` manages feature generation, lags, and recursive forecasting.
- **Algorithm**: The core predictive model is `LGBMRegressor` (LightGBM).
- **Time Series Features**:
  - **Lags**: Extensive historical lags are introduced (e.g., 1 to 168 hours).
  - **Rolling Transformations**: Rolling statistics (Mean, Std, Min, Max) are calculated over 24-hour and 168-hour windows to capture daily and weekly profiles.
  - **Temporal Features**: Includes hour of day, day of week, day of year, month, and month boundary indicators.
- **Hyperparameter Optimization**: Optional `optuna` integration for Bayesian tuning. It employs Time Series Cross Validation with 4 expanding windows (predicting 720 hours ahead), optimized against Mean Absolute Error (MAE).
- **Robust Defaults**: Without optimization, robust LightGBM defaults are applied (e.g., `n_estimators=400`, `learning_rate=0.03`, deep trees with strong regularization).

### 3. Forecasting & Aggregation
- **Horizon Generation**: The model predicts hourly forecasts approximately 1 year into the future to ensure coverage through the October 2025 target.
- **Future Exogenous Variables**: Future static variables (`deviceType`, `region`) remain constant. Future dynamic features are imputed using historical averages per device and hour of the day, with fallbacks to overall feature means.
- **Aggregation**: Final hourly predictions (`LGBMRegressor` output) are aggregated into monthly means per device. The results are filtered to match the required submission timeframe (May 2025 - October 2025) and formatted to match the expected submission schema: `deviceId`, `year`, `month`, `prediction`.