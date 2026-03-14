import polars as pl
from loguru import logger
from datetime import datetime
from sklearn.cluster import KMeans

def load_data(data_path: str) -> pl.LazyFrame:
    logger.info("Setting up lazy data processing...")
    try:
        devices_path = data_path.replace("data.csv", "devices.csv")
        lazy_df = pl.scan_csv(data_path)
        
        # Load devices eagerly for clustering
        devices_df = pl.read_csv(devices_path)
        
        # Perform KMeans clustering
        coords = devices_df.select(["latitude", "longitude"]).to_numpy()
        n_clusters = min(16, len(devices_df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        regions = kmeans.fit_predict(coords)
        
        # Add region column and drop raw coordinates
        devices_df = devices_df.with_columns(pl.Series("region", regions))
        devices_df = devices_df.drop(["latitude", "longitude"])
        
        # Join with main lazy frame
        lazy_df = lazy_df.join(devices_df.lazy(), on="deviceId", how="left")
        
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
        .drop(["period"], strict=False)
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
            pl.col("x2").sum().alias("x2_mean"),
            pl.all().exclude(["deviceId", "timedate", "Hour_Start", "x2"]).mean()
        )
        .with_columns(
            pl.all().exclude(["deviceId", "Hour_Start", "x2_mean", "region", "deviceType"])
            .mean()
            .over(["region", "Hour_Start"])
            .name.prefix("region_")
        )
    )

    df = processed_lazy.collect()
    
    logger.info("Segmenting devices based on historical consumption...")
    # Calculate historical consumption stats per device
    device_stats = df.group_by("deviceId").agg(
        pl.col("x2_mean").mean().alias("x2_mean_avg"),
        pl.col("x2_mean").std().alias("x2_mean_std")
    ).fill_null(0)
    
    # Perform KMeans clustering for consumption segmentation
    n_cons_clusters = min(8, len(device_stats))
    X_cons = device_stats.select(["x2_mean_avg", "x2_mean_std"]).to_numpy()
    kmeans_cons = KMeans(n_clusters=n_cons_clusters, random_state=42, n_init="auto")
    segments = kmeans_cons.fit_predict(X_cons)
    
    device_stats = device_stats.with_columns(pl.Series("consumption_segment", segments))
    
    # Join the segment back to the main DataFrame
    df = df.join(device_stats.select(["deviceId", "consumption_segment"]), on="deviceId", how="left")
    
    logger.info(f"Processed DataFrame shape: {df.shape}")
    logger.debug(f"\n{df.head()}")
    return df
