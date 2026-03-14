import polars as pl
import pandas as pd
from loguru import logger
import numpy as np

def run_baseline_model(df: pl.DataFrame) -> pd.DataFrame:
    logger.info("Running alternative historical scaling model (baseline)...")
    pdf = df.select(["deviceId", "Hour_Start", "x2_mean"]).to_pandas()
    
    # Ensure Datetime
    pdf['Hour_Start'] = pd.to_datetime(pdf['Hour_Start'])
    
    # 1. Past 6 months: Nov 2024 - Apr 2025
    mask_recent = (pdf['Hour_Start'] >= '2024-11-01') & (pdf['Hour_Start'] < '2025-05-01')
    recent_6m = pdf[mask_recent].groupby('deviceId')['x2_mean'].mean()
    
    # 2. Previous 6 months (year prior): Nov 2023 - Apr 2024
    mask_prior = (pdf['Hour_Start'] >= '2023-11-01') & (pdf['Hour_Start'] < '2024-05-01')
    prior_6m = pdf[mask_prior].groupby('deviceId')['x2_mean'].mean()
    
    # Compute factor
    factors = (recent_6m / prior_6m).fillna(1.0).replace([np.inf, -np.inf], 1.0)
    # Clip factors to reasonable range to avoid extreme outliers
    factors = factors.clip(lower=0.5, upper=2.0)
    
    # 3. Values from May 2024 to Oct 2024
    mask_target_hist = (pdf['Hour_Start'] >= '2024-05-01') & (pdf['Hour_Start'] < '2024-11-01')
    target_hist = pdf[mask_target_hist].copy()
    target_hist['month'] = target_hist['Hour_Start'].dt.month
    
    # Aggregate to monthly mean per device
    monthly_hist = target_hist.groupby(['deviceId', 'month'])['x2_mean'].mean().reset_index()
    
    # Scale values
    monthly_hist['factor'] = monthly_hist['deviceId'].map(factors).fillna(1.0)
    monthly_hist['prediction'] = monthly_hist['x2_mean'] * monthly_hist['factor']
    
    # Format output
    monthly_hist['year'] = 2025
    submission_df = monthly_hist[['deviceId', 'year', 'month', 'prediction']].copy()
    
    # Handle missing months for all devices
    import itertools
    all_devices = pdf['deviceId'].unique()
    expected_months = list(range(5, 11))
    
    # Create full grid of devices and months
    grid = pd.DataFrame(list(itertools.product(all_devices, [2025], expected_months)), columns=['deviceId', 'year', 'month'])
    
    # Merge existing predictions
    submission_df = pd.merge(grid, submission_df, on=['deviceId', 'year', 'month'], how='left')
    
    # Fill missing predictions
    if submission_df['prediction'].isna().any():
        logger.info(f"Filling missing predictions for {submission_df['prediction'].isna().sum()} device-month pairs.")
        global_mean = pdf['x2_mean'].mean()
        fallback_map = recent_6m.to_dict()
        
        def fill_missing(row):
            if pd.isna(row['prediction']):
                return fallback_map.get(row['deviceId'], global_mean)
            return row['prediction']
            
        submission_df['prediction'] = submission_df.apply(fill_missing, axis=1)
        submission_df['prediction'] = submission_df['prediction'].fillna(global_mean)

    submission_df = submission_df.sort_values(['deviceId', 'month']).reset_index(drop=True)
    
    logger.info(f"Baseline model generated {len(submission_df)} predictions.")
    return submission_df