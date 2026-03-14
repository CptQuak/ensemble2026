import polars as pl
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd, RollingMin, RollingMax
from lightgbm import LGBMRegressor
from loguru import logger
import optuna
from sklearn.metrics import mean_absolute_error
import numpy as np


def create_mlforecast_model(params: dict) -> MLForecast:
    return MLForecast(
        models={"LGBMRegressor": LGBMRegressor(**params)},
        freq="h",
        lags=[1, 2, 3, 4, 6, 12, 24, 48, 72, 168],
        lag_transforms={
            1: [
                RollingMean(window_size=24),
                RollingMean(window_size=168),
                RollingStd(window_size=24),
                RollingMin(window_size=24),
                RollingMax(window_size=24),
            ],
            24: [RollingMean(window_size=168)],
        },
        date_features=[
            "month",
            "hour",
            "dayofweek",
            "dayofyear",
            "is_month_start",
            "is_month_end",
        ],
    )


def train_model(
    df: pl.DataFrame, optimize: bool = False, n_trials: int = 20
) -> MLForecast:
    logger.info(f"Setting up forecasting pipeline (Optimize={optimize})...")
    forecast_df = df.rename(
        {"deviceId": "unique_id", "Hour_Start": "ds", "x2_mean": "y"}
    )

    forecast_df = forecast_df.sort(["unique_id", "ds"])

    # mlforecast takes a pandas dataframe by default
    forecast_df_pd = forecast_df.to_pandas()

    # Ensure there are no missing periods in the time series (required by MLForecast cross_validation)
    forecast_df_pd = (
        forecast_df_pd.set_index("ds")
        .groupby("unique_id")
        .resample("1h")
        .ffill()
        .drop(columns="unique_id", errors="ignore")
        .reset_index()
    )
    # Forward fill handles missing inner values, but we also ensure trailing/leading nans are filled if any
    for col in forecast_df_pd.columns:
        if col not in ["unique_id", "ds"]:
            forecast_df_pd[col] = forecast_df_pd.groupby("unique_id")[col].bfill()

    if optimize:

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "random_state": 42,
                "verbose": -1,
                "n_jobs": 4,  # Utilize all cores for training
            }

            mlf = create_mlforecast_model(params)

            static_cols = ["deviceType", "region"]
            static_cols = [c for c in static_cols if c in forecast_df_pd.columns]

            try:
                # Time series CV: 4 windows (more robust than 2), predicting 720 hours ahead
                cv_res = mlf.cross_validation(
                    df=forecast_df_pd,
                    h=720,
                    n_windows=4,
                    step_size=720,
                    static_features=static_cols,
                )
                # Calculate MAE
                mae = mean_absolute_error(cv_res["y"], cv_res["LGBMRegressor"])
                return mae
            except Exception as e:
                logger.warning(f"Trial failed during cross validation: {e}")
                return float("inf")

        logger.info(f"Starting Optuna optimization with {n_trials} trials...")
        # Suppress verbose optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, n_jobs=4)

        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["verbose"] = -1
        best_params["n_jobs"] = -1
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best Validation MAE: {study.best_value}")
    else:
        # Default robust hyperparameters if optimize is False

        best_params = {
            "n_estimators": 400,  # Increased capacity since we have many new features
            "learning_rate": 0.03,  # A standard, robust learning rate for this many estimators
            "max_depth": 6,  # Deeper trees to capture interactions between lags, regions, and exogenous vars
            "num_leaves": 31,  # Standard default; prevents overfitting while respecting max_depth (31 < 2^6)
            "min_child_samples": 50,  # Increased to heavily regularize and prevent overfitting on specific noisy timestamps
            "colsample_bytree": 0.8,  # IMPORTANT: Randomly samples 80% of features per tree. Crucial when you have many highly correlated lag/rolling features.
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,  # Utilize all cores for training
        }
        logger.info(f"Using default parameters: {best_params}")

    logger.info("Training final model with best parameters...")
    final_mlf = create_mlforecast_model(best_params)

    static_cols = ["deviceType", "region"]
    static_cols = [c for c in static_cols if c in forecast_df_pd.columns]

    try:
        final_mlf.fit(forecast_df_pd, static_features=static_cols)
        return final_mlf
    except Exception as e:
        logger.error(f"Error during final model fitting: {e}")
        raise
