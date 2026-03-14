import polars as pl
import pandas as pd
from prophet import Prophet
from loguru import logger
import numpy as np

def run_prophet_model(df: pl.DataFrame) -> pd.DataFrame:
    logger.info("Initializing Prophet forecasting...")
    
    # 1. Prepare historical dataframe
    train_df = df.rename(
        {"deviceId": "unique_id", "Hour_Start": "ds", "x2_mean": "y"}
    ).to_pandas()
    
    # Prophet requires log scale for multiplicative seasonality or to handle strictly positive skewed targets
    train_df['y'] = np.log1p(train_df['y'])
    
    device_ids = train_df['unique_id'].unique()
    logger.info(f"Training separate Prophet models for {len(device_ids)} devices.")
    
    all_predictions = []
    
    # We need to predict until the end of October 2025.
    # The training data goes up to October 2024.
    # To cover May-Oct 2025, we need about a year of forecasts.
    h_hours = 365 * 24 + 10 * 24 # ~8900 hours, we do 9000 to be safe.
    
    for idx, device_id in enumerate(device_ids):
        if idx % 5 == 0:
            logger.info(f"Processing device {idx + 1}/{len(device_ids)}: {device_id}")
            
        device_df = train_df[train_df['unique_id'] == device_id][['ds', 'y']]
        
        # Ensure 'ds' is strictly datetime and no missing values in target
        device_df = device_df.dropna(subset=['y'])
        
        # Initialize Prophet with strong seasonality priors
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Disable logging for each individual fit
        import logging
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        
        try:
            m.fit(device_df)
            
            future = m.make_future_dataframe(periods=h_hours, freq='h')
            forecast = m.predict(future)
            
            # Extract just the needed columns and invert the log transform
            forecast = forecast[['ds', 'yhat']].copy()
            forecast['yhat'] = np.expm1(forecast['yhat'])
            forecast['unique_id'] = device_id
            
            all_predictions.append(forecast)
        except Exception as e:
            logger.error(f"Prophet failed for device {device_id}: {e}")
            continue
            
    if not all_predictions:
        raise ValueError("Prophet failed for all devices.")
        
    logger.info("Aggregating Prophet predictions...")
    predictions = pd.concat(all_predictions, ignore_index=True)
    
    # Format according to example_submission.py
    # deviceId, year, month, prediction
    
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
    
    # Rename unique_id to deviceId and yhat to prediction
    submission_df = submission_df.rename(columns={
        'unique_id': 'deviceId',
        'yhat': 'prediction'
    })
    
    # Final column selection and ordering
    submission_df = submission_df[['deviceId', 'year', 'month', 'prediction']]
    
    logger.info(f"Final submission format generated with shape {submission_df.shape}")
    logger.debug(f"\n{submission_df.head()}")
    
    return submission_df
