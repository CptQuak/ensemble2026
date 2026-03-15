"""Monthly cross-sectional regression model for device energy consumption."""
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Columns that are not features
_NON_FEATURE_COLS = {"deviceId", "year", "month", "period", "x2_monthly_mean"}

# Categorical columns (integer-encoded, passed as categoricals to LGB)
_CAT_COLS = ["deviceType", "region", "consumption_segment"]


def _get_feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def _get_temp_mean_cols(feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if c.endswith("_mean") and c.startswith("t")]


def _to_numpy(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select(cols).to_numpy().astype(np.float64)


def lomocv_validate(train_monthly_df: pl.DataFrame) -> float:
    """Leave-one-month-out cross-validation. Returns mean MAE across months."""
    months = sorted(train_monthly_df.select(["year", "month"]).unique().rows())
    feature_cols = _get_feature_cols(train_monthly_df)
    temp_mean_cols = _get_temp_mean_cols(feature_cols)

    maes = []
    naive_maes = []

    for year, month in months:
        mask_val = (pl.col("year") == year) & (pl.col("month") == month)
        fold_train = train_monthly_df.filter(~mask_val)
        fold_val = train_monthly_df.filter(mask_val)

        if len(fold_train) == 0 or len(fold_val) == 0:
            continue

        X_tr = _to_numpy(fold_train, feature_cols)
        y_tr = fold_train["x2_monthly_mean"].to_numpy()
        X_val = _to_numpy(fold_val, feature_cols)
        y_val = fold_val["x2_monthly_mean"].to_numpy()

        lgb = LGBMRegressor(
            objective="mae",
            max_depth=5,
            reg_alpha=1.0,
            reg_lambda=2.0,
            colsample_bytree=0.7,
            min_child_samples=20,
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        cat_idx = [i for i, c in enumerate(feature_cols) if c in _CAT_COLS]
        lgb.fit(X_tr, y_tr, categorical_feature=cat_idx if cat_idx else "auto")

        scaler = StandardScaler()
        X_tr_ridge = scaler.fit_transform(_to_numpy(fold_train, temp_mean_cols))
        X_val_ridge = scaler.transform(_to_numpy(fold_val, temp_mean_cols))
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr_ridge, y_tr)

        pred_lgb = lgb.predict(X_val)
        pred_ridge = ridge.predict(X_val_ridge)
        pred = 0.7 * pred_lgb + 0.3 * pred_ridge
        pred = np.clip(pred, 0.0, None)

        mae = mean_absolute_error(y_val, pred)
        maes.append(mae)

        # Naive: per-device training mean
        device_means = fold_train.group_by("deviceId").agg(
            pl.col("x2_monthly_mean").mean().alias("dev_mean")
        )
        val_with_mean = fold_val.join(device_means, on="deviceId", how="left").with_columns(
            pl.col("dev_mean").fill_null(y_tr.mean())
        )
        naive_pred = val_with_mean["dev_mean"].to_numpy()
        naive_mae = mean_absolute_error(y_val, naive_pred)
        naive_maes.append(naive_mae)

        logger.info(
            f"  LOMOCV {year}-{month:02d}: MAE={mae:.4f}  naive={naive_mae:.4f}  "
            f"{'BEAT' if mae < naive_mae else 'MISS'}"
        )

    mean_mae = float(np.mean(maes)) if maes else float("nan")
    mean_naive = float(np.mean(naive_maes)) if naive_maes else float("nan")
    logger.info(
        f"LOMOCV mean MAE: {mean_mae:.4f}  naive: {mean_naive:.4f}  "
        f"({'BEAT' if mean_mae < mean_naive else 'MISS'} naive baseline)"
    )
    return mean_mae


def train_monthly_model(
    train_monthly_df: pl.DataFrame,
) -> tuple[LGBMRegressor, Ridge, list[str], list[str], StandardScaler]:
    """Train LightGBM + Ridge ensemble on monthly device-level training data.

    Returns
    -------
    lgb_model, ridge_model, feature_cols, temp_mean_cols, scaler
    """
    feature_cols = _get_feature_cols(train_monthly_df)
    temp_mean_cols = _get_temp_mean_cols(feature_cols)

    X = _to_numpy(train_monthly_df, feature_cols)
    y = train_monthly_df["x2_monthly_mean"].to_numpy()

    logger.info(
        f"Training monthly model: {len(train_monthly_df)} rows, {len(feature_cols)} features"
    )

    lgb_model = LGBMRegressor(
        objective="mae",
        max_depth=5,
        reg_alpha=1.0,
        reg_lambda=2.0,
        colsample_bytree=0.7,
        min_child_samples=20,
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    cat_idx = [i for i, c in enumerate(feature_cols) if c in _CAT_COLS]
    lgb_model.fit(X, y, categorical_feature=cat_idx if cat_idx else "auto")

    scaler = StandardScaler()
    X_ridge = scaler.fit_transform(_to_numpy(train_monthly_df, temp_mean_cols))
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_ridge, y)

    logger.info("Monthly model training complete.")
    return lgb_model, ridge_model, feature_cols, temp_mean_cols, scaler


def predict_monthly(
    lgb_model: LGBMRegressor,
    ridge_model: Ridge,
    feature_cols: list[str],
    temp_mean_cols: list[str],
    scaler: StandardScaler,
    forecast_monthly_df: pl.DataFrame,
) -> pd.DataFrame:
    """Generate submission-format predictions for forecast months.

    Returns a pandas DataFrame with columns: deviceId, year, month, prediction
    """
    if len(forecast_monthly_df) == 0:
        logger.warning("forecast_monthly_df is empty — returning empty submission.")
        return pd.DataFrame(columns=["deviceId", "year", "month", "prediction"])

    X = _to_numpy(forecast_monthly_df, feature_cols)
    X_ridge = scaler.transform(_to_numpy(forecast_monthly_df, temp_mean_cols))

    pred_lgb = lgb_model.predict(X)
    pred_ridge = ridge_model.predict(X_ridge)
    pred = 0.7 * pred_lgb + 0.3 * pred_ridge
    pred = np.clip(pred, 0.0, None)

    submission = pd.DataFrame(
        {
            "deviceId": forecast_monthly_df["deviceId"].to_list(),
            "year": forecast_monthly_df["year"].to_list(),
            "month": forecast_monthly_df["month"].to_list(),
            "prediction": pred,
        }
    )

    # Keep only May–Oct 2025 (months 5–10)
    submission = submission[
        (submission["year"] == 2025)
        & (submission["month"] >= 5)
        & (submission["month"] <= 10)
    ].copy()

    submission = submission.sort_values(["deviceId", "year", "month"]).reset_index(drop=True)
    logger.info(f"Submission shape: {submission.shape}")
    logger.debug(f"\n{submission.head()}")
    return submission
