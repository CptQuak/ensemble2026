import os
import argparse
from datetime import datetime
import polars as pl
import pandas as pd
from loguru import logger
import numpy as np
import window_ops.rolling
from sklearn.metrics import mean_absolute_error
import optuna
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from prophet import Prophet
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

from src.data_processing import load_data

def process_monthly(lazy_df: pl.LazyFrame) -> pl.DataFrame:
    logger.info("Aggregating data monthly...")
    processed_lazy = (
        lazy_df.with_columns(
            pl.col("timedate").str.strip_suffix(" UTC").str.to_datetime()
        )
        .drop(["period"], strict=False)
        .filter(pl.col("timedate") < datetime(2025, 5, 1))
        .sort(["deviceId", "timedate"])
        .with_columns(
            pl.all().exclude(["deviceId", "timedate"])
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .over("deviceId")
        )
        .with_columns(pl.col("timedate").dt.truncate("1mo").alias("Month_Start"))
        .group_by(["deviceId", "Month_Start"])
        .agg(
            pl.col("x2").mean().alias("x2_mean"),
            pl.all().exclude(["deviceId", "timedate", "Month_Start", "x2"]).mean()
        )
        .with_columns(
            pl.all().exclude(["deviceId", "Month_Start", "x2_mean", "region", "deviceType"])
            .mean()
            .over(["region", "Month_Start"])
            .name.prefix("region_")
        )
    )

    df = processed_lazy.collect()
    
    # Calculate historical consumption stats per device (using the monthly data)
    device_stats = df.group_by("deviceId").agg(
        pl.col("x2_mean").mean().alias("x2_mean_avg"),
        pl.col("x2_mean").std().alias("x2_mean_std")
    ).fill_null(0)
    
    from sklearn.cluster import KMeans
    n_cons_clusters = min(8, len(device_stats))
    X_cons = device_stats.select(["x2_mean_avg", "x2_mean_std"]).to_numpy()
    kmeans_cons = KMeans(n_clusters=n_cons_clusters, random_state=42, n_init="auto")
    segments = kmeans_cons.fit_predict(X_cons)
    
    device_stats = device_stats.with_columns(pl.Series("consumption_segment", segments))
    df = df.join(device_stats.select(["deviceId", "consumption_segment"]), on="deviceId", how="left")
    
    logger.info(f"Monthly DataFrame shape: {df.shape}")
    return df

def train_and_forecast_monthly_lgbm(df: pl.DataFrame, artifacts_dir: str, validate: bool = False, val_months: int = 2, optimize: bool = False, n_trials: int = 20):
    logger.info(f"Setting up monthly LightGBM pipeline (Validate={validate}, Optimize={optimize}, Horizon={val_months} months)...")
    forecast_df = df.rename(
        {"deviceId": "unique_id", "Month_Start": "ds", "x2_mean": "y"}
    ).to_pandas()

    forecast_df = forecast_df.sort_values(["unique_id", "ds"])

    # Extract static features before resampling
    static_cols = ["deviceType", "region", "consumption_segment"]
    static_cols = [c for c in static_cols if c in forecast_df.columns]
    static_features = forecast_df[["unique_id"] + static_cols].drop_duplicates("unique_id")

    forecast_df = (
        forecast_df.set_index("ds")
        .groupby("unique_id")
        .resample("MS")
        .mean(numeric_only=True)
        .drop(columns="unique_id", errors="ignore")
        .reset_index()
    )
    
    forecast_df = forecast_df.sort_values(["unique_id", "ds"])
    for col in forecast_df.columns:
        if col not in ["unique_id", "ds"]:
            forecast_df[col] = forecast_df.groupby("unique_id")[col].ffill().bfill()
            
    forecast_df = forecast_df.drop(columns=[c for c in static_cols if c in forecast_df.columns])
    forecast_df = forecast_df.merge(static_features, on="unique_id", how="left")

    forecast_df['y'] = np.log1p(forecast_df['y'])

    def create_monthly_mlf(params):
        return MLForecast(
            models={"LGBMRegressor": LGBMRegressor(**params)},
            freq="MS",
            lags=[1, 2],
            lag_transforms={
                1: [(window_ops.rolling.rolling_mean, 2)]
            },
            date_features=["month", "year", "quarter"],
        )

    if optimize:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1
            }

            mlf_opt = create_monthly_mlf(params)
            
            max_len = forecast_df.groupby('unique_id').size().max()
            # Need max_len >= val_months * n_windows + max_lag + 1
            if max_len < val_months * 2 + 3:
                n_windows = 1
                if max_len < val_months + 3:
                    logger.warning(f"Dataset too small for optimization CV (max {max_len} months). Need {val_months + 3}. Returning inf.")
                    return float('inf')
            else:
                n_windows = 2
                
            try:
                cv_res = mlf_opt.cross_validation(
                    df=forecast_df,
                    h=val_months,
                    n_windows=n_windows,
                    static_features=static_cols
                )
                cv_res['y'] = np.expm1(cv_res['y'])
                cv_res['LGBMRegressor'] = np.expm1(cv_res['LGBMRegressor'])
                mae = mean_absolute_error(cv_res["y"], cv_res["LGBMRegressor"])
                return mae
            except Exception as e:
                logger.warning(f"Trial failed during cross validation: {e}")
                return float("inf")

        logger.info(f"Starting Optuna optimization with {n_trials} trials...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, n_jobs=4, show_progress_bar=True)

        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["verbose"] = -1
        best_params["n_jobs"] = -1
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best Validation MAE: {study.best_value}")
    else:
        best_params = {
            "n_estimators": 150, 
            "learning_rate": 0.05, 
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42, 
            "verbose": -1,
            "n_jobs": -1
        }
        logger.info(f"Using default parameters: {best_params}")

    mlf = create_monthly_mlf(best_params)

    if validate and not optimize:
        max_len = forecast_df.groupby('unique_id').size().max()
        required_len = val_months + 3 # val_months + 2 lags + 1 training sample
        if max_len < required_len:
            logger.warning(f"Dataset too small for validation (max {max_len} months per device). Need at least {required_len}. Skipping validation.")
        else:
            logger.info(f"Running cross-validation on the last {val_months} months...")
            try:
                cv_res = mlf.cross_validation(
                    df=forecast_df,
                    h=val_months,
                    n_windows=1,
                    static_features=static_cols
                )
                cv_res['y'] = np.expm1(cv_res['y'])
                cv_res['LGBMRegressor'] = np.expm1(cv_res['LGBMRegressor'])
                mae = mean_absolute_error(cv_res["y"], cv_res["LGBMRegressor"])
                logger.info(f"Validation MAE (Monthly, last {val_months} months): {mae}")
            except Exception as e:
                logger.error(f"Error during validation: {e}")

    logger.info("Training final monthly LightGBM model...")
    mlf.fit(forecast_df, static_features=static_cols)
    h_months = 12
    
    dyn_cols = [c for c in forecast_df.columns if c not in ['unique_id', 'ds', 'y'] + static_cols]
    
    if dyn_cols:
        X_df = mlf.make_future_dataframe(h=h_months)
        forecast_df['month'] = forecast_df['ds'].dt.month
        hist_avg = forecast_df.groupby(['unique_id', 'month'])[dyn_cols].mean().reset_index()
        X_df['month'] = X_df['ds'].dt.month
        X_df = X_df.merge(hist_avg, on=['unique_id', 'month'], how='left')
        
        device_mean = forecast_df.groupby('unique_id')[dyn_cols].mean()
        overall_mean = forecast_df[dyn_cols].mean()
        for col in dyn_cols:
            X_df[col] = X_df[col].fillna(X_df['unique_id'].map(device_mean[col])).fillna(overall_mean[col])
            
        X_df = X_df.drop(columns=['month'])
        predictions = mlf.predict(h=h_months, X_df=X_df)
    else:
        predictions = mlf.predict(h=h_months)

    predictions['prediction'] = np.expm1(predictions['LGBMRegressor'])
    return predictions

def train_and_forecast_monthly_prophet(df: pl.DataFrame, validate: bool = False, val_months: int = 2):
    logger.info(f"Setting up monthly Prophet pipeline (Validate={validate}, Horizon={val_months} months)...")
    train_df = df.rename(
        {"deviceId": "unique_id", "Month_Start": "ds", "x2_mean": "y"}
    ).to_pandas()
    
    train_df['y'] = np.log1p(train_df['y'])
    device_ids = train_df['unique_id'].unique()
    
    all_predictions = []
    val_maes = []
    h_months = 12
    
    for device_id in device_ids:
        device_df = train_df[train_df['unique_id'] == device_id][['ds', 'y']].dropna().sort_values('ds')
        
        if validate:
            if len(device_df) <= val_months + 1:
                logger.warning(f"Device {device_id} does not have enough data for validation (needs >{val_months + 1} months). Skipping validation.")
                fit_df = device_df
                run_val = False
            else:
                fit_df = device_df.iloc[:-val_months]
                val_df = device_df.iloc[-val_months:]
                run_val = True
        else:
            fit_df = device_df
            run_val = False
            
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        
        import logging
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        
        try:
            m.fit(fit_df)
            
            if run_val:
                future_val = m.make_future_dataframe(periods=val_months, freq='MS')
                forecast_val = m.predict(future_val)
                preds_val = forecast_val.tail(val_months)['yhat'].values
                actuals_val = val_df['y'].values
                
                mae = mean_absolute_error(np.expm1(actuals_val), np.expm1(preds_val))
                val_maes.append(mae)
                
                # Refit on all data
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                m.fit(device_df)
                
            future = m.make_future_dataframe(periods=h_months, freq='MS')
            forecast = m.predict(future)
            
            res = forecast[['ds', 'yhat']].tail(h_months).copy()
            res['prediction'] = np.expm1(res['yhat'])
            res['unique_id'] = device_id
            all_predictions.append(res)
        except Exception as e:
            logger.error(f"Prophet failed for device {device_id}: {e}")
            
    if validate and val_maes:
        logger.info(f"Validation MAE (Monthly, last {val_months} months) across valid devices: {np.mean(val_maes)}")
            
    return pd.concat(all_predictions, ignore_index=True)

def train_and_forecast_monthly_nbeats(df: pl.DataFrame, validate: bool = False, val_months: int = 2):
    logger.info(f"Setting up monthly NBEATS pipeline (Validate={validate}, Horizon={val_months} months)...")
    train_df = df.rename(
        {"deviceId": "unique_id", "Month_Start": "ds", "x2_mean": "y"}
    ).to_pandas()
    
    train_df['ds'] = pd.to_datetime(train_df['ds']).astype('datetime64[ns]')
    train_df['y'] = np.log1p(train_df['y'])
    
    # We only need to predict up to October 2025 (which is 6 months from May)
    h_months = 6
    
    if validate:
        logger.info(f"Running NBEATS validation on the last {val_months} months...")
        min_len = train_df.groupby('unique_id').size().min()
        if min_len <= val_months + 1:
            logger.warning(f"Dataset too small for validation (min length {min_len}). Skipping validation.")
        else:
            fit_df = train_df.groupby('unique_id').head(-val_months).reset_index(drop=True)
            val_df = train_df.groupby('unique_id').tail(val_months).reset_index(drop=True)
            
            model_val = NBEATS(h=val_months, input_size=1, max_steps=100)
            nf_val = NeuralForecast(models=[model_val], freq='MS')
            
            # Use supress warnings and logs
            import logging
            logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
            
            nf_val.fit(df=fit_df)
            forecast_val = nf_val.predict()
            
            if 'unique_id' not in forecast_val.columns:
                forecast_val = forecast_val.reset_index(names='unique_id')
                
            merged = forecast_val.merge(val_df[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'])
            mae = mean_absolute_error(np.expm1(merged['y']), np.expm1(merged['NBEATS']))
            logger.info(f"Validation MAE (Monthly, last {val_months} months) NBEATS: {mae}")
            
    logger.info("Training final monthly NBEATS model...")
    model_final = NBEATS(h=h_months, input_size=1, max_steps=100)
    nf_final = NeuralForecast(models=[model_final], freq='MS')
    
    import logging
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    
    nf_final.fit(df=train_df)
    predictions = nf_final.predict()
    
    if 'unique_id' not in predictions.columns:
        predictions = predictions.reset_index(names='unique_id')
        
    predictions['prediction'] = np.expm1(predictions['NBEATS'])
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Run the monthly time series forecasting pipeline.")
    parser.add_argument("--data_path", type=str, default="data/data.csv", help="Path to the input CSV data.")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to store artifacts and logs.")
    parser.add_argument("--model", type=str, choices=["lgbm", "prophet", "nbeats"], default="lgbm", help="Choose the model to run.")
    parser.add_argument("--validate", action="store_true", help="Run validation on the hold-out set.")
    parser.add_argument("--val_months", type=int, default=2, help="Number of months to hold out for validation.")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna hyperparameter optimization before training.")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials.")
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    logger.add(os.path.join(args.artifacts_dir, "monthly_pipeline_{time}.log"), rotation="10 MB")

    logger.info(f"Starting execution of monthly pipeline (Model={args.model})")
    lazy_df = load_data(args.data_path)
    df = process_monthly(lazy_df)
    
    if args.model == "lgbm":
        predictions = train_and_forecast_monthly_lgbm(df, args.artifacts_dir, validate=args.validate, val_months=args.val_months, optimize=args.optimize, n_trials=args.n_trials)
    elif args.model == "prophet":
        predictions = train_and_forecast_monthly_prophet(df, validate=args.validate, val_months=args.val_months)
    elif args.model == "nbeats":
        predictions = train_and_forecast_monthly_nbeats(df, validate=args.validate, val_months=args.val_months)
    
    # Format output
    predictions['year'] = predictions['ds'].dt.year
    predictions['month'] = predictions['ds'].dt.month
    
    submission_df = predictions[
        (predictions['year'] == 2025) & (predictions['month'] >= 5) & (predictions['month'] <= 10)
    ].copy()
    
    submission_df = submission_df.rename(columns={'unique_id': 'deviceId'})
    submission_df = submission_df[['deviceId', 'year', 'month', 'prediction']]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.artifacts_dir, f"run_monthly_{args.model}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    predictions_path = os.path.join(run_dir, "predictions.csv")
    submission_df.to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main()
