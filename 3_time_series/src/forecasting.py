import pandas as pd
from mlforecast import MLForecast
from loguru import logger

def generate_forecasts(mlf: MLForecast, h: int = 6) -> pd.DataFrame:
    h_hours = h * 30 * 24
    logger.info(f"Generating hourly forecasts for {h_hours} hours (approx {h} months)...")
    
    predictions = mlf.predict(h=h_hours)
    logger.info(f"Hourly predictions generated with shape {predictions.shape}")
    
    logger.info("Aggregating hourly forecasts to monthly averages...")
    # Convert hourly `ds` back to the start of the month
    predictions['Month_Start'] = pd.to_datetime(predictions['ds'].dt.strftime('%Y-%m-01'))
    
    # Group by `unique_id` and `Month_Start` and average the predictions
    monthly_predictions = (
        predictions
        .groupby(['unique_id', 'Month_Start'])
        .mean(numeric_only=True)
        .reset_index()
    )
    
    # Drop original ds, rename Month_Start to ds to maintain structure
    if 'ds' in monthly_predictions.columns:
        monthly_predictions = monthly_predictions.drop(columns=['ds'])
    monthly_predictions = monthly_predictions.rename(columns={'Month_Start': 'ds'})
    
    logger.info(f"Monthly predictions aggregated with shape {monthly_predictions.shape}")
    logger.debug(f"\n{monthly_predictions.head()}")
    return monthly_predictions

