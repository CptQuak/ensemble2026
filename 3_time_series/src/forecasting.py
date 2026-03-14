import pandas as pd
from mlforecast import MLForecast
from loguru import logger

def generate_forecasts(mlf: MLForecast, h: int = 6) -> pd.DataFrame:
    logger.info(f"Generating forecasts for {h} months...")
    predictions = mlf.predict(h=h)
    logger.info(f"Predictions generated with shape {predictions.shape}")
    logger.debug(f"\n{predictions.head()}")
    return predictions
