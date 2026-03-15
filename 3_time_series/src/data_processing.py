import math
import numpy as np
import polars as pl
from loguru import logger
from sklearn.cluster import KMeans


def load_data(data_path: str) -> pl.LazyFrame:
    logger.info("Setting up lazy data processing...")
    try:
        devices_path = data_path.replace("data.csv", "devices.csv")
        lazy_df = pl.scan_csv(data_path)

        # Load devices eagerly for clustering
        devices_df = pl.read_csv(devices_path)

        # Perform KMeans clustering on coordinates
        coords = devices_df.select(["latitude", "longitude"]).to_numpy()
        n_clusters = min(16, len(devices_df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        regions = kmeans.fit_predict(coords)

        # Keep latitude/longitude for downstream features; add region
        devices_df = devices_df.with_columns(pl.Series("region", regions))

        # Join with main lazy frame (period column preserved)
        lazy_df = lazy_df.join(devices_df.lazy(), on="deviceId", how="left")

        return lazy_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def _daylight_hours_vec(lat_deg: np.ndarray, month: np.ndarray) -> np.ndarray:
    """Vectorised approximate daylight hours (15th of each month)."""
    month_to_doy = np.array([0, 15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    doy = month_to_doy[month]
    decl = 23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81)))
    lat_r = np.radians(lat_deg)
    decl_r = np.radians(decl)
    cos_ha = -np.tan(lat_r) * np.tan(decl_r)
    cos_ha = np.clip(cos_ha, -1.0, 1.0)
    ha = np.degrees(np.arccos(cos_ha))
    return 2.0 * ha / 15.0


def compute_monthly_features(lazy_df: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Aggregate raw 5-min data to device × month statistics.

    Returns
    -------
    train_monthly_df : pl.DataFrame   – rows where period == 'train'
    forecast_monthly_df : pl.DataFrame – rows where period in ['valid', 'test']
    """
    logger.info("Computing monthly features...")

    temp_cols = [f"t{i}" for i in range(1, 14)]

    # Build aggregation expressions
    agg_exprs: list[pl.Expr] = []

    for col in temp_cols:
        agg_exprs += [
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_std"),
            pl.col(col).min().alias(f"{col}_min"),
            pl.col(col).max().alias(f"{col}_max"),
            pl.col(col).quantile(0.25).alias(f"{col}_q25"),
            pl.col(col).quantile(0.75).alias(f"{col}_q75"),
        ]

    agg_exprs += [
        pl.col("x1").mean().alias("x1_mean"),
        pl.col("x1").std().alias("x1_std"),
        pl.col("x1").min().alias("x1_min"),
        pl.col("x1").max().alias("x1_max"),
        pl.col("x1").quantile(0.25).alias("x1_q25"),
        pl.col("x1").quantile(0.75).alias("x1_q75"),
        (pl.col("x1") == 0).mean().alias("x1_zero_frac"),
        pl.col("x3").mean().alias("x3_mean"),
        pl.col("x3").std().alias("x3_std"),
        pl.col("x3").min().alias("x3_min"),
        pl.col("x3").max().alias("x3_max"),
        pl.col("x3").quantile(0.25).alias("x3_q25"),
        pl.col("x3").quantile(0.75).alias("x3_q75"),
        pl.col("x2").mean().alias("x2_monthly_mean"),
        # Device-level constants
        pl.col("deviceType").first(),
        pl.col("region").first(),
        pl.col("latitude").first(),
        pl.col("longitude").first(),
        pl.col("period").first(),
    ]

    monthly_lazy = (
        lazy_df
        .with_columns(
            pl.col("timedate").str.strip_suffix(" UTC").str.to_datetime()
        )
        .sort(["deviceId", "timedate"])
        .with_columns(
            pl.all().exclude(["deviceId", "timedate", "period"])
            .forward_fill()
            .backward_fill()
            .over("deviceId")
        )
        .with_columns(
            pl.col("timedate").dt.year().alias("year"),
            pl.col("timedate").dt.month().cast(pl.Int32).alias("month"),
        )
        .group_by(["deviceId", "year", "month"])
        .agg(agg_exprs)
    )

    df = monthly_lazy.collect()
    logger.info(f"Monthly aggregation shape: {df.shape}")

    # --- Cross-sensor stats (derived after collect) ---
    t_mean_cols = [f"{c}_mean" for c in temp_cols]
    t_means_arr = df.select(t_mean_cols).to_numpy()
    df = df.with_columns(
        pl.Series("t_mean_of_means", t_means_arr.mean(axis=1)),
        pl.Series("t_std_of_means", t_means_arr.std(axis=1)),
    )

    # --- Temporal features ---
    month_arr = df["month"].to_numpy()
    lat_arr = df["latitude"].to_numpy()
    df = df.with_columns(
        pl.Series("month_sin", np.sin(2 * math.pi * month_arr / 12)),
        pl.Series("month_cos", np.cos(2 * math.pi * month_arr / 12)),
        pl.Series("daylight_hours", _daylight_hours_vec(lat_arr, month_arr)),
    )

    # --- Historical per-device stats (from training rows only) ---
    train_rows = df.filter(pl.col("period") == "train")

    if len(train_rows) > 0:
        device_hist = (
            train_rows
            .group_by("deviceId")
            .agg(
                pl.col("x2_monthly_mean").mean().alias("x2_mean_avg"),
                pl.col("x2_monthly_mean").std().fill_null(0.0).alias("x2_mean_std"),
            )
        )

        # Consumption segmentation via KMeans
        n_seg = min(8, len(device_hist))
        X_seg = device_hist.select(["x2_mean_avg", "x2_mean_std"]).to_numpy()
        seg_labels = KMeans(n_clusters=n_seg, random_state=42, n_init="auto").fit_predict(X_seg)
        device_hist = device_hist.with_columns(
            pl.Series("consumption_segment", seg_labels)
        )

        df = df.join(device_hist, on="deviceId", how="left")
        # For forecast rows without training history, fill with 0
        df = df.with_columns(
            pl.col("x2_mean_avg").fill_null(0.0),
            pl.col("x2_mean_std").fill_null(0.0),
            pl.col("consumption_segment").fill_null(0),
        )
    else:
        # Subsample-only scenario: no training data
        df = df.with_columns(
            pl.lit(0.0).alias("x2_mean_avg"),
            pl.lit(0.0).alias("x2_mean_std"),
            pl.lit(0).alias("consumption_segment"),
        )

    # --- Fill remaining nulls in feature columns ---
    feature_num_cols = [
        c for c in df.columns
        if c not in ["deviceId", "year", "month", "period", "x2_monthly_mean", "deviceType",
                     "region", "consumption_segment"]
        and df[c].dtype in (pl.Float64, pl.Float32, pl.Int32, pl.Int64)
    ]
    df = df.with_columns(
        [pl.col(c).fill_null(0.0) for c in feature_num_cols]
    )

    train_monthly = df.filter(pl.col("period") == "train")
    forecast_monthly = df.filter(pl.col("period").is_in(["valid", "test"]))

    logger.info(f"Train monthly shape: {train_monthly.shape}, Forecast monthly shape: {forecast_monthly.shape}")
    return train_monthly, forecast_monthly


def process_and_aggregate(lazy_df: pl.LazyFrame) -> pl.DataFrame:
    """Legacy hourly aggregation (used by mlforecast, neural, prophet pipelines)."""
    from datetime import datetime
    logger.info("Aggregating data hourly...")
    processed_lazy = (
        lazy_df.with_columns(
            pl.col("timedate").str.strip_suffix(" UTC").str.to_datetime()
        )
        .drop(["period", "latitude", "longitude"], strict=False)
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
        .with_columns(
            pl.all().exclude(["deviceId", "Hour_Start", "x2_mean", "region", "deviceType"])
            .mean()
            .over(["region", "Hour_Start"])
            .name.prefix("region_")
        )
    )

    df = processed_lazy.collect()

    logger.info("Segmenting devices based on historical consumption...")
    device_stats = df.group_by("deviceId").agg(
        pl.col("x2_mean").mean().alias("x2_mean_avg"),
        pl.col("x2_mean").std().alias("x2_mean_std")
    ).fill_null(0)

    n_cons_clusters = min(8, len(device_stats))
    X_cons = device_stats.select(["x2_mean_avg", "x2_mean_std"]).to_numpy()
    kmeans_cons = KMeans(n_clusters=n_cons_clusters, random_state=42, n_init="auto")
    segments = kmeans_cons.fit_predict(X_cons)

    device_stats = device_stats.with_columns(pl.Series("consumption_segment", segments))
    df = df.join(device_stats.select(["deviceId", "consumption_segment"]), on="deviceId", how="left")

    logger.info(f"Processed DataFrame shape: {df.shape}")
    logger.debug(f"\n{df.head()}")
    return df
