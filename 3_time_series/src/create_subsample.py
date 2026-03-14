import polars as pl
from loguru import logger
import os

def create_subsample(data_path: str, devices_path: str, output_data_path: str, output_devices_path: str):
    logger.info("Starting subsampling process...")
    
    # Load a subsample of 5 devices
    logger.info("Loading 5 devices for subsample...")
    try:
        devices_df = pl.read_csv(devices_path)
        test_devices_df = devices_df.head(5)
        test_devices = test_devices_df["deviceId"].to_list()
        logger.info(f"Test devices selected: {test_devices}")
        
        # Save subsampled devices
        test_devices_df.write_csv(output_devices_path)
        logger.success(f"Saved subsampled devices to {output_devices_path}")
    except Exception as e:
        logger.error(f"Failed to load or save devices: {e}")
        return

    # Lazy load data.csv and filter to test devices
    logger.info("Setting up lazy data processing for subsampling...")
    try:
        lazy_df = pl.scan_csv(data_path)
        subsample_data_df = lazy_df.filter(pl.col("deviceId").is_in(test_devices)).collect()
        
        subsample_data_df.write_csv(output_data_path)
        logger.success(f"Saved subsampled data to {output_data_path} with shape {subsample_data_df.shape}")
    except Exception as e:
        logger.error(f"Failed to process and save data subsample: {e}")
        return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a subsample of the dataset.")
    parser.add_argument("--data_path", type=str, default="data/data.csv", help="Path to the original data CSV.")
    parser.add_argument("--devices_path", type=str, default="data/devices.csv", help="Path to the devices CSV.")
    parser.add_argument("--output_data_path", type=str, default="data/subsample_data.csv", help="Path to save the subsampled data CSV.")
    parser.add_argument("--output_devices_path", type=str, default="data/subsample_devices.csv", help="Path to save the subsampled devices CSV.")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory for logs.")
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    logger.add(os.path.join(args.artifacts_dir, "subsample_{time}.log"), rotation="10 MB")
    
    create_subsample(args.data_path, args.devices_path, args.output_data_path, args.output_devices_path)
