import polars as pl
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from loguru import logger
import os
from datetime import datetime


def run_pipeline(data_path: str, devices_path: str, artifacts_dir: str):
    logger.info("Starting data processing pipeline...")

    # 1. Load a subsample of 5 devices
    logger.info("Loading 5 devices for testing...")
    try:
        devices_df = pl.read_csv(devices_path)
        test_devices = devices_df.head(5)["deviceId"].to_list()
        logger.info(f"Test devices selected: {test_devices}")
    except Exception as e:
        logger.error(f"Failed to load devices: {e}")
        return

    # 2. Lazy load data.csv and filter to test devices
    logger.info("Setting up lazy data processing...")
    lazy_df = pl.scan_csv(data_path)
    lazy_df = lazy_df.filter(pl.col("deviceId").is_in(test_devices))
    # 3. Process and aggregate monthly
    logger.info("Aggregating data monthly...")
    processed_lazy = lazy_df.with_columns(
        pl.col("timedate").str.strip_suffix(" UTC").str.to_datetime()
    ).sort(
        ["deviceId", "timedate"]
    ).with_columns(
        pl.col("x2").fill_null(strategy="forward").fill_null(strategy="backward").over("deviceId")
    ).with_columns(
        pl.col("timedate").dt.truncate("1mo").alias("Month_Start")
    ).group_by(
        ["deviceId", "Month_Start"]
    ).agg(
        pl.col("x2").mean().alias("x2_mean")
    )

    df = processed_lazy.collect()
    logger.info(f"Processed DataFrame shape: {df.shape}")
    logger.debug(f"\n{df.head()}")

    # 4. Forecasting Pipeline (mlforecast)
    logger.info("Setting up forecasting pipeline...")
    forecast_df = df.rename(
        {"deviceId": "unique_id", "Month_Start": "ds", "x2_mean": "y"}
    )

    forecast_df = forecast_df.sort(["unique_id", "ds"])

    # mlforecast takes a pandas dataframe by default
    forecast_df_pd = forecast_df.to_pandas()

    mlf = MLForecast(
        models=[LGBMRegressor(random_state=42, n_estimators=50, verbose=-1)],
        freq="MS",  # 'MS' for Month Start
        lags=[1],
    )

    logger.info("Fitting LGBM model...")
    try:
        mlf.fit(forecast_df_pd)
    except Exception as e:
        logger.error(f"Error during model fitting: {e}")
        return

    logger.info("Generating forecasts for 6 months...")
    predictions = mlf.predict(h=6)
    logger.info(f"Predictions generated with shape {predictions.shape}")
    logger.debug(f"\n{predictions.head()}")

    # Save artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(artifacts_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    predictions_path = os.path.join(run_dir, "predictions.csv")
    predictions.to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to {predictions_path}")
