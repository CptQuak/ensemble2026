import pandas as pd
from mlforecast import MLForecast
from loguru import logger

def generate_forecasts(mlf: MLForecast, h: int = 6) -> pd.DataFrame:
    # 6 months x 30 days x 24 hours (approx)
    # We need to predict until the end of October 2025.
    # The training data goes up to October 2024.
    # To cover May-Oct 2025, we need about a year of forecasts.
    h_hours = 365 * 24 + 10 * 24 # Over-forecast to be safe
    logger.info(f"Generating hourly forecasts for {h_hours} hours...")
    
    predictions = mlf.predict(h=h_hours)
    
    # Format according to example_submission.py
    # deviceId, year, month, prediction
    
    logger.info("Formatting forecasts to required submission format (deviceId, year, month, prediction)...")
    
    predictions['year'] = predictions['ds'].dt.year
    predictions['month'] = predictions['ds'].dt.month
    
    # Filter for May to October 2025
    submission_df = predictions[
        (predictions['year'] == 2025) & 
        (predictions['month'] >= 5) & 
        (predictions['month'] <= 10)
    ].copy()
    
    # Aggregate hourly to monthly mean
    submission_df = (
        submission_df
        .groupby(['unique_id', 'year', 'month'])
        .mean(numeric_only=True)
        .reset_index()
    )
    
    # Rename unique_id to deviceId and LGBMRegressor to prediction
    submission_df = submission_df.rename(columns={
        'unique_id': 'deviceId',
        'LGBMRegressor': 'prediction'
    })
    
    # Final column selection and ordering
    submission_df = submission_df[['deviceId', 'year', 'month', 'prediction']]
    
    logger.info(f"Final submission format generated with shape {submission_df.shape}")
    logger.debug(f"\n{submission_df.head()}")
    return submission_df

