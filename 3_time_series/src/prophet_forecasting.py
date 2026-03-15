import polars as pl
import pandas as pd
from prophet import Prophet
from loguru import logger
import numpy as np
from sklearn.metrics import mean_absolute_error

def run_prophet_model(df: pl.DataFrame, validate: bool = False) -> pd.DataFrame:
    logger.info(f"Initializing Prophet forecasting (Validate={validate})...")
    
    # 1. Prepare historical dataframe
    train_df = df.rename(
        {"deviceId": "unique_id", "Hour_Start": "ds", "x2_mean": "y"}
    ).to_pandas()
    
    # Prophet requires log scale for multiplicative seasonality or to handle strictly positive skewed targets
    train_df['y'] = np.log1p(train_df['y'])
    
    device_ids = train_df['unique_id'].unique()
    logger.info(f"Training separate Prophet models for {len(device_ids)} devices.")
    
    all_predictions = []
    val_maes = []
    
    # We need to predict until the end of October 2025.
    # The training data goes up to October 2024.
    # To cover May-Oct 2025, we need about a year of forecasts.
    h_hours = 365 * 24 + 10 * 24 # ~8900 hours, we do 9000 to be safe.
    
    # 6 months is approx 184 days = 4416 hours
    val_hours = 4416
    
    for idx, device_id in enumerate(device_ids):
        if idx % 5 == 0:
            logger.info(f"Processing device {idx + 1}/{len(device_ids)}: {device_id}")
            
        device_df = train_df[train_df['unique_id'] == device_id][['ds', 'y']]
        
        # Ensure 'ds' is strictly datetime and no missing values in target
        device_df = device_df.dropna(subset=['y'])
        device_df = device_df.sort_values('ds')
        
        if validate:
            if len(device_df) <= val_hours:
                logger.warning(f"Device {device_id} does not have enough data for validation. Skipping validation for it.")
                fit_df = device_df
                run_val = False
            else:
                fit_df = device_df.iloc[:-val_hours]
                val_df = device_df.iloc[-val_hours:]
                run_val = True
        else:
            fit_df = device_df
            run_val = False
        
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
            m.fit(fit_df)
            
            if run_val:
                future_val = m.make_future_dataframe(periods=val_hours, freq='h')
                forecast_val = m.predict(future_val)
                # Match predictions to validation set
                preds_val = forecast_val.tail(val_hours)['yhat'].values
                actuals_val = val_df['y'].values
                
                # Reverse log transformation for MAE calculation
                preds_val_expm1 = np.expm1(preds_val)
                actuals_val_expm1 = np.expm1(actuals_val)
                
                mae = mean_absolute_error(actuals_val_expm1, preds_val_expm1)
                val_maes.append(mae)
                
                # Refit on all data if validate is True, because we still want to generate final forecasts
                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    changepoint_prior_scale=0.05
                )
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
            
    if validate and val_maes:
        logger.info(f"Validation MAE (Hourly, last 6 months) across valid devices: {np.mean(val_maes)}")
            
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
