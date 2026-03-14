import polars as pl
from loguru import logger
from datetime import datetime

def load_data(data_path: str) -> pl.LazyFrame:
    logger.info("Setting up lazy data processing...")
    try:
        lazy_df = pl.scan_csv(data_path)
        return lazy_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def process_and_aggregate(lazy_df: pl.LazyFrame) -> pl.DataFrame:
    logger.info("Aggregating data hourly...")
    processed_lazy = (
        lazy_df.with_columns(
            pl.col("timedate").str.strip_suffix(" UTC").str.to_datetime()
        )
        .filter(pl.col("timedate") < datetime(2025, 5, 1))
        .sort(["deviceId", "timedate"])
        .with_columns(
            pl.all().exclude(["deviceId", "timedate"])
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .over("deviceId")
        )
        .with_columns(pl.col("timedate").dt.truncate("1h").alias("Hour_Start"))
        .group_by(["deviceId", "Hour_Start"])
        .agg(
            pl.col("x2").mean().alias("x2_mean"),
            pl.all().exclude(["deviceId", "timedate", "Hour_Start", "x2"]).mean()
        )
    )

    df = processed_lazy.collect()
    logger.info(f"Processed DataFrame shape: {df.shape}")
    logger.debug(f"\n{df.head()}")
    return df
