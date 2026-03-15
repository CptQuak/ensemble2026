import polars as pl
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from loguru import logger
import sys

# Suppress loguru logs from data processing to keep output clean
logger.remove()
logger.add(sys.stdout, level="WARNING")

from src.data_processing import load_data, process_and_aggregate

def main():
    df = process_and_aggregate(load_data('data/subsample_data.csv'))
    pdf = df.select(["deviceId", "Hour_Start", "x2_mean"]).to_pandas()
    pdf['Hour_Start'] = pd.to_datetime(pdf['Hour_Start'])
    
    # Validation Target: 2024-11-01 to 2025-05-01
    mask_target = (pdf['Hour_Start'] >= '2024-11-01') & (pdf['Hour_Start'] < '2025-05-01')
    target_df = pdf[mask_target].copy()
    target_df['month'] = target_df['Hour_Start'].dt.month
    actuals = target_df.groupby(['deviceId', 'month'])['x2_mean'].mean().reset_index()
    actuals.rename(columns={'x2_mean': 'actual'}, inplace=True)
    
    # Learn Factor: 2024-05-01 to 2024-11-01 vs 2023-05-01 to 2023-11-01
    mask_recent_factor = (pdf['Hour_Start'] >= '2024-05-01') & (pdf['Hour_Start'] < '2024-11-01')
    mask_prior_factor = (pdf['Hour_Start'] >= '2023-05-01') & (pdf['Hour_Start'] < '2023-11-01')
    
    recent_factor_mean = pdf[mask_recent_factor].groupby('deviceId')['x2_mean'].mean()
    prior_factor_mean = pdf[mask_prior_factor].groupby('deviceId')['x2_mean'].mean()
    
    factors = (recent_factor_mean / prior_factor_mean).fillna(1.0).replace([np.inf, -np.inf], 1.0)
    factors = factors.clip(lower=0.5, upper=2.0)
    
    # Base values: 2023-11-01 to 2024-05-01
    mask_base = (pdf['Hour_Start'] >= '2023-11-01') & (pdf['Hour_Start'] < '2024-05-01')
    base_df = pdf[mask_base].copy()
    base_df['month'] = base_df['Hour_Start'].dt.month
    
    base_monthly = base_df.groupby(['deviceId', 'month'])['x2_mean'].mean().reset_index()
    
    # Apply factor
    base_monthly['factor'] = base_monthly['deviceId'].map(factors).fillna(1.0)
    base_monthly['prediction'] = base_monthly['x2_mean'] * base_monthly['factor']
    
    # Join predictions with actuals
    merged = pd.merge(actuals, base_monthly[['deviceId', 'month', 'prediction']], on=['deviceId', 'month'], how='left')
    
    # Fill any missing predictions with global mean
    global_mean = pdf['x2_mean'].mean()
    merged['prediction'] = merged['prediction'].fillna(global_mean)
    
    mae = mean_absolute_error(merged['actual'], merged['prediction'])
    print(f"\n--- Evaluation Results ---")
    print(f"Validation MAE (Monthly, Nov 2024 - Apr 2025): {mae:.6f}")
    
if __name__ == "__main__":
    main()
