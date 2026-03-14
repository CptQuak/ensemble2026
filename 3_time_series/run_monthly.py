import os
import argparse
from datetime import datetime
import polars as pl
import pandas as pd
from loguru import logger
import numpy as np
from sklearn.metrics import mean_absolute_error
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from prophet import Prophet

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

def train_and_forecast_monthly_lgbm(df: pl.DataFrame, artifacts_dir: str, validate: bool = False):
    logger.info(f"Setting up monthly LightGBM pipeline (Validate={validate})...")
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

    mlf = MLForecast(
        models={
            "LGBMRegressor": LGBMRegressor(
                n_estimators=100, 
                learning_rate=0.05, 
                max_depth=4, 
                random_state=42, 
                verbose=-1
            )
        },
        freq="MS",
        lags=[1, 2],
        date_features=["month", "year"],
    )

    if validate:
        logger.info("Running cross-validation on the last 6 months...")
        try:
            cv_res = mlf.cross_validation(
                df=forecast_df,
                h=6,
                n_windows=1,
                static_features=static_cols
            )
            cv_res['y'] = np.expm1(cv_res['y'])
            cv_res['LGBMRegressor'] = np.expm1(cv_res['LGBMRegressor'])
            mae = mean_absolute_error(cv_res["y"], cv_res["LGBMRegressor"])
            logger.info(f"Validation MAE (Monthly, last 6 months): {mae}")
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

def train_and_forecast_monthly_prophet(df: pl.DataFrame, validate: bool = False):
    logger.info(f"Setting up monthly Prophet pipeline (Validate={validate})...")
    train_df = df.rename(
        {"deviceId": "unique_id", "Month_Start": "ds", "x2_mean": "y"}
    ).to_pandas()
    
    train_df['y'] = np.log1p(train_df['y'])
    device_ids = train_df['unique_id'].unique()
    
    all_predictions = []
    val_maes = []
    h_months = 12
    val_months = 6
    
    for device_id in device_ids:
        device_df = train_df[train_df['unique_id'] == device_id][['ds', 'y']].dropna().sort_values('ds')
        
        if validate:
            if len(device_df) <= val_months + 1:
                logger.warning(f"Device {device_id} does not have enough data for validation. Skipping validation.")
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
        logger.info(f"Validation MAE (Monthly, last 6 months) across valid devices: {np.mean(val_maes)}")
            
    return pd.concat(all_predictions, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description="Run the monthly time series forecasting pipeline.")
    parser.add_argument("--data_path", type=str, default="data/subsample_data.csv", help="Path to the input CSV data.")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to store artifacts and logs.")
    parser.add_argument("--model", type=str, choices=["lgbm", "prophet"], default="lgbm", help="Choose the model to run.")
    parser.add_argument("--validate", action="store_true", help="Run validation on the last 6 months of the training set.")
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    logger.add(os.path.join(args.artifacts_dir, "monthly_pipeline_{time}.log"), rotation="10 MB")

    logger.info(f"Starting execution of monthly pipeline (Model={args.model})")
    lazy_df = load_data(args.data_path)
    df = process_monthly(lazy_df)
    
    if args.model == "lgbm":
        predictions = train_and_forecast_monthly_lgbm(df, args.artifacts_dir, validate=args.validate)
    else:
        predictions = train_and_forecast_monthly_prophet(df, validate=args.validate)
    
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
