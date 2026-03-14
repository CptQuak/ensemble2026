import os
from loguru import logger
from src.pipeline import run_pipeline


def main():
    # Configure Logguru
    os.makedirs("artifacts", exist_ok=True)
    logger.add("artifacts/pipeline_{time}.log", rotation="10 MB")

    data_path = "data/data.csv"
    devices_path = "data/devices.csv"
    artifacts_dir = "artifacts"

    logger.info("Starting execution of main.py")
    run_pipeline(data_path, devices_path, artifacts_dir)
    logger.success("Execution completed successfully.")


if __name__ == "__main__":
    main()
