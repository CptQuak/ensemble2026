import polars as pl
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from loguru import logger

def train_model(df: pl.DataFrame) -> MLForecast:
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
        return mlf
    except Exception as e:
        logger.error(f"Error during model fitting: {e}")
        raise
