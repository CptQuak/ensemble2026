import pandas as pd
import polars as pl
import numpy as np
from loguru import logger
from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR, NBEATS, NBEATSx

def prepare_neural_data(df: pl.DataFrame) -> pd.DataFrame:
    forecast_df = df.rename(
        {"deviceId": "unique_id", "Hour_Start": "ds", "x2_mean": "y"}
    )
    forecast_df = forecast_df.sort(["unique_id", "ds"])
    forecast_df_pd = forecast_df.to_pandas()
    
    # Label encode categorical features with sorting for consistency
    for col in ["deviceType", "region"]:
        if col in forecast_df_pd.columns:
            forecast_df_pd[col] = pd.factorize(forecast_df_pd[col], sort=True)[0]
    
    # Fill missing gaps
    # We group by unique_id, then resample the datetime index (ds)
    forecast_df_pd = forecast_df_pd.set_index("ds")
    resampled_list = []
    for uid, group in forecast_df_pd.groupby("unique_id"):
        resampled = group.resample("1h").ffill().bfill()
        resampled["unique_id"] = uid
        resampled_list.append(resampled)
    
    forecast_df_pd = pd.concat(resampled_list).reset_index()
    
    # Normalize y with log1p like in MLForecast
    forecast_df_pd['y'] = np.log1p(forecast_df_pd['y'].astype(float))
    
    # Add hour column if not exist for simple feature
    forecast_df_pd['hour'] = forecast_df_pd['ds'].dt.hour
    
    return forecast_df_pd

def train_neural_model(df: pl.DataFrame, model_name: str, optimize: bool = False, validate: bool = False) -> NeuralForecast:
    logger.info(f"Setting up NeuralForecast pipeline for model: {model_name}")
    forecast_df_pd = prepare_neural_data(df)
    
    static_cols = ["deviceType", "region"]
    static_cols = [c for c in static_cols if c in forecast_df_pd.columns]
    
    dyn_cols = [c for c in forecast_df_pd.columns if c not in ['unique_id', 'ds', 'y'] + static_cols]
    
    # We need to predict approx 6 months (May-Oct 2025)
    h = 4416 
    
    # Reduced input_size to ensure it fits in shorter datasets (e.g. subsample)
    input_size = 168 # 1 week
    
    unique_ids = forecast_df_pd['unique_id'].nunique()
    rows_per_id = len(forecast_df_pd) / unique_ids
    
    # Ensure we have at least one window for training
    if rows_per_id < (h + input_size):
        logger.warning(f"Data is too short for horizon {h}. Reducing h.")
        h = int(rows_per_id - input_size - 1)
        if h < 24: h = 24
        
    # Validation size
    val_size = h if validate and (rows_per_id > 2*h + input_size) else 0
    if validate and val_size == 0:
        logger.warning("Data too short for validation. Skipping validation split.")
    
    models = []
    all_exog = dyn_cols + static_cols
    
    # Early stopping only if validation is used
    patience = 3 if val_size > 0 else -1
    
    if model_name == "deepar":
        models.append(DeepAR(h=h, input_size=input_size,
                             futr_exog_list=all_exog,
                             lstm_hidden_size=64, # Reduced from 100
                             max_steps=300, early_stop_patience_steps=patience))
    elif model_name == "nbeats":
        models.append(NBEATS(h=h, input_size=input_size,
                             stack_types=['identity', 'trend', 'seasonality'],
                             n_blocks=[1, 1, 1],
                             mlp_units=[[256, 256], [256, 256], [256, 256]],
                             batch_size=4, windows_batch_size=16,
                             max_steps=300, early_stop_patience_steps=patience))
    elif model_name == "nbeatsx":
        models.append(NBEATSx(h=h, input_size=input_size,
                              futr_exog_list=all_exog,
                              stack_types=['identity', 'trend', 'seasonality'],
                              n_blocks=[1, 1, 1],
                              mlp_units=[[256, 256], [256, 256], [256, 256]],
                              batch_size=4, windows_batch_size=16,
                              max_steps=300, early_stop_patience_steps=patience))
    else:
        raise ValueError(f"Unsupported model: {model_name}")
                              
    nf = NeuralForecast(models=models, freq='h')
    
    logger.info(f"Training {model_name} with h={h}, val_size={val_size}...")
    nf.fit(df=forecast_df_pd, val_size=val_size)
    
    return nf

def generate_neural_forecasts(nf: NeuralForecast, df: pl.DataFrame) -> pd.DataFrame:
    # Target is Oct 2025. 
    # The models are trained with a specific h, so we use that.
    h = nf.models[0].h
    logger.info(f"Generating hourly forecasts for {h} hours...")
    
    forecast_df_pd = prepare_neural_data(df)
    
    static_cols = ["deviceType", "region"]
    static_cols = [c for c in static_cols if c in forecast_df_pd.columns]
    dyn_cols = [c for c in forecast_df_pd.columns if c not in ['unique_id', 'ds', 'y'] + static_cols]
    
    # Create future exogenous variables
    future_dates = pd.date_range(start=forecast_df_pd['ds'].max() + pd.Timedelta(hours=1), periods=h, freq='h')
    unique_ids = forecast_df_pd['unique_id'].unique()
    
    future_df = pd.DataFrame({
        'unique_id': np.repeat(unique_ids, len(future_dates)),
        'ds': np.tile(future_dates, len(unique_ids))
    })
    
    # Add static features
    static_mapping = forecast_df_pd[['unique_id'] + static_cols].drop_duplicates()
    future_df = future_df.merge(static_mapping, on='unique_id', how='left')
    
    if dyn_cols:
        future_df['hour'] = future_df['ds'].dt.hour
        hist_avg = forecast_df_pd.groupby(['unique_id', 'hour'])[dyn_cols].mean().reset_index()
        future_df = future_df.merge(hist_avg, on=['unique_id', 'hour'], how='left')
        
        device_mean = forecast_df_pd.groupby('unique_id')[dyn_cols].mean()
        overall_mean = forecast_df_pd[dyn_cols].mean()
        
        for col in dyn_cols:
            device_mean_map = future_df['unique_id'].map(device_mean[col])
            future_df[col] = future_df[col].fillna(device_mean_map)
            future_df[col] = future_df[col].fillna(overall_mean[col])
            
        future_df = future_df.drop(columns=['hour'])
        
    logger.info(f"Predicting...")
    predictions = nf.predict(df=forecast_df_pd, futr_df=future_df)
        
    predictions = predictions.reset_index()
    if 'unique_id' not in predictions.columns:
        predictions = predictions.reset_index(names='unique_id')
        
    model_col = [col for col in predictions.columns if col not in ['unique_id', 'ds']][0]
    predictions[model_col] = np.expm1(predictions[model_col])
    
    logger.info("Formatting forecasts to required submission format...")

    predictions['year'] = predictions['ds'].dt.year
    predictions['month'] = predictions['ds'].dt.month
    
    submission_df = predictions[
        (predictions['year'] == 2025) & 
        (predictions['month'] >= 5) & 
        (predictions['month'] <= 10)
    ].copy()
    
    if submission_df.empty:
        logger.warning("No forecasts found in the target period (May-Oct 2025). Check h and date range.")
    
    submission_df = (
        submission_df
        .groupby(['unique_id', 'year', 'month'])
        .mean(numeric_only=True)
        .reset_index()
    )
    
    submission_df = submission_df.rename(columns={
        'unique_id': 'deviceId',
        model_col: 'prediction'
    })
    
    submission_df = submission_df[['deviceId', 'year', 'month', 'prediction']]
    
    logger.info(f"Final submission format generated with shape {submission_df.shape}")
    return submission_df
