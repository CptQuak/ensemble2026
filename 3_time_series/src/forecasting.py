import pandas as pd
import polars as pl
from mlforecast import MLForecast
from loguru import logger

def generate_forecasts(mlf: MLForecast, df: pl.DataFrame, h: int = 6) -> pd.DataFrame:
    # 6 months x 30 days x 24 hours (approx)
    # We need to predict until the end of October 2025.
    # The training data goes up to October 2024.
    # To cover May-Oct 2025, we need about a year of forecasts.
    h_hours = 365 * 24 + 10 * 24 # Over-forecast to be safe
    logger.info(f"Generating hourly forecasts for {h_hours} hours...")
    
    # 1. Prepare historical dataframe for exogenous variables
    train_df = df.rename(
        {"deviceId": "unique_id", "Hour_Start": "ds", "x2_mean": "y"}
    ).to_pandas()
    
    static_cols = ['deviceType', 'region']
    static_cols = [c for c in static_cols if c in train_df.columns]
    
    exclude_features = ['x1', 'x3']
    dyn_cols = [c for c in train_df.columns if c not in ['unique_id', 'ds', 'y'] + static_cols + exclude_features]
    
    if dyn_cols or static_cols:
        logger.info(f"Creating future exogenous variables. Dynamic: {dyn_cols}, Static: {static_cols}")
        X_df = mlf.make_future_dataframe(h=h_hours)
        
        if dyn_cols:
            train_df['hour'] = train_df['ds'].dt.hour
            hist_avg = train_df.groupby(['unique_id', 'hour'])[dyn_cols].mean().reset_index()
            
            X_df['hour'] = X_df['ds'].dt.hour
            X_df = X_df.merge(hist_avg, on=['unique_id', 'hour'], how='left')
            
            device_mean = train_df.groupby('unique_id')[dyn_cols].mean()
            overall_mean = train_df[dyn_cols].mean()
            
            for col in dyn_cols:
                device_mean_map = X_df['unique_id'].map(device_mean[col])
                X_df[col] = X_df[col].fillna(device_mean_map)
                X_df[col] = X_df[col].fillna(overall_mean[col])
                
            X_df = X_df.drop(columns=['hour'])
            
        predictions = mlf.predict(h=h_hours, X_df=X_df)
    else:
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

