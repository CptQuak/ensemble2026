import os
from datetime import datetime
from loguru import logger

from src.data_processing import load_data, process_and_aggregate
from src.model_training import train_model
from src.forecasting import generate_forecasts
from src.baseline import run_baseline_model

def run_pipeline(data_path: str, artifacts_dir: str, optimize: bool = False, validate: bool = False, model_type: str = "mlforecast"):
    logger.info("Starting data processing pipeline...")

    # 1. Data Processing
    try:
        lazy_df = load_data(data_path)
        df = process_and_aggregate(lazy_df)
    except Exception as e:
        logger.error(f"Pipeline failed during data processing: {e}")
        return

    if model_type == "baseline":
        try:
            predictions = run_baseline_model(df)
        except Exception as e:
            logger.error(f"Pipeline failed during baseline forecasting: {e}")
            return
    else:
        # 2. Model Training
        try:
            mlf = train_model(df, optimize=optimize, validate=validate)
        except Exception as e:
            logger.error(f"Pipeline failed during model training: {e}")
            return

        # 3. Forecasting
        try:
            predictions = generate_forecasts(mlf, df, h=6)
        except Exception as e:
            logger.error(f"Pipeline failed during forecasting: {e}")
            return

    # Save artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(artifacts_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    predictions_path = os.path.join(run_dir, "predictions.csv")
    predictions.to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to {predictions_path}")
