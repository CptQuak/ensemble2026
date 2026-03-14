import os
import argparse
from loguru import logger
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the time series forecasting pipeline.")
    parser.add_argument("--data_path", type=str, default="data/subsample_data.csv", help="Path to the input CSV data.")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to store artifacts and logs.")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna hyperparameter optimization before training.")
    parser.add_argument("--validate", action="store_true", help="Run validation on the last 6 months of the training set.")
    args = parser.parse_args()

    # Configure Logguru
    os.makedirs(args.artifacts_dir, exist_ok=True)
    logger.add(os.path.join(args.artifacts_dir, "pipeline_{time}.log"), rotation="10 MB")

    logger.info("Starting execution of main.py")
    run_pipeline(args.data_path, args.artifacts_dir, optimize=args.optimize, validate=args.validate)
    logger.success("Execution completed successfully.")


if __name__ == "__main__":
    main()
