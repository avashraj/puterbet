"""models_ml/train.py

Train one XGBoost regressor per stat category (points, rebounds, assists, etc.)
using walk-forward-safe features from the feature engine.

Usage:
    uv run python -m models_ml.train                    # P0 stats only
    uv run python -m models_ml.train --all-stats        # P0 + P1 stats
    uv run python -m models_ml.train --stats points assists
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_FEATURES = Path(__file__).parents[1] / "data" / "features.parquet"
_SAVED_DIR = Path(__file__).parent / "saved"

# --- Column classifications ---

IDENTITY_COLS = ["player_id", "game_id", "team_id", "date", "season"]

TARGET_COLS = [
    "points", "rebounds", "assists", "three_pm", "steals", "blocks",
    "turnovers", "minutes_played",
]

# Raw box score columns that are targets or leaked info (same-game stats)
LEAKAGE_COLS = [
    "fgm", "fga", "three_pa", "ftm", "fta", "pf", "plus_minus",
    "minutes",  # raw string column
]

# Columns that shouldn't be features (intermediate/helper columns)
EXCLUDE_COLS = [
    "dnp", "date_dt", "playoff_round", "home_team_id", "away_team_id", "starter",
]

# --- Season splits ---

TRAIN_SEASONS = ["2022-23", "2023-24"]
VAL_SEASON = "2024-25"
TEST_SEASON = "2025-26"

# --- Stat priorities ---

P0_STATS = ["points", "rebounds", "assists"]
P1_STATS = ["three_pm", "steals", "blocks"]

# --- Default XGBoost hyperparameters ---

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Dynamically compute the list of safe feature columns.

    Excludes identity, target, leakage, helper, and string columns.
    Returns a sorted list for reproducibility.
    """
    exclude = set(IDENTITY_COLS + TARGET_COLS + LEAKAGE_COLS + EXCLUDE_COLS)
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == object:
            continue
        feature_cols.append(col)
    return sorted(feature_cols)


def load_features(path: Path | str = _DEFAULT_FEATURES) -> pd.DataFrame:
    """Load the feature parquet and filter to played rows (dnp == 0)."""
    path = Path(path)
    log.info("Loading features from %s", path)
    df = pd.read_parquet(path)
    n_total = len(df)
    df = df[df["dnp"] == 0].reset_index(drop=True)
    log.info("Loaded %d rows (%d DNP rows excluded)", len(df), n_total - len(df))
    return df


def split_data(
    df: pd.DataFrame,
    stat: str,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data by season into train/val/test numpy arrays.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    train_mask = df["season"].isin(TRAIN_SEASONS)
    val_mask = df["season"] == VAL_SEASON
    test_mask = df["season"] == TEST_SEASON

    X_train = df.loc[train_mask, feature_cols].values.astype(np.float32)
    y_train = df.loc[train_mask, stat].values.astype(np.float32)
    X_val = df.loc[val_mask, feature_cols].values.astype(np.float32)
    y_val = df.loc[val_mask, stat].values.astype(np.float32)
    X_test = df.loc[test_mask, feature_cols].values.astype(np.float32)
    y_test = df.loc[test_mask, stat].values.astype(np.float32)

    log.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(y_train), len(y_val), len(y_test),
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(
    stat: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
) -> tuple[XGBRegressor, dict]:
    """Train an XGBRegressor for a single stat with early stopping.

    Returns (model, metadata_dict).
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, stat, feature_cols)

    log.info("Training XGBoost for '%s' ...", stat)
    model = XGBRegressor(**params, early_stopping_rounds=30)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Predictions and MAE on val set
    val_pred = model.predict(X_val)
    val_mae = float(np.mean(np.abs(y_val - val_pred)))

    # Baseline: rolling mean 10 on val set
    baseline_col = f"{stat}_roll_mean_10"
    val_df = df[df["season"] == VAL_SEASON]
    if baseline_col in val_df.columns:
        baseline_pred = val_df[baseline_col].values.astype(np.float32)
        # Use only rows where baseline is not NaN for fair comparison
        valid = ~np.isnan(baseline_pred)
        baseline_mae = float(np.mean(np.abs(y_val[valid] - baseline_pred[valid])))
        # Also compute model MAE on same valid subset
        model_mae_valid = float(np.mean(np.abs(y_val[valid] - val_pred[valid])))
    else:
        baseline_mae = float("nan")
        model_mae_valid = val_mae

    improvement = (
        (baseline_mae - model_mae_valid) / baseline_mae * 100
        if not np.isnan(baseline_mae) and baseline_mae > 0
        else float("nan")
    )

    # Top 20 features by importance
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    top_features = [(feature_cols[i], float(importances[i])) for i in top_idx]

    metadata = {
        "stat": stat,
        "feature_cols": feature_cols,
        "params": params,
        "val_mae": val_mae,
        "val_mae_valid_subset": model_mae_valid,
        "baseline_mae": baseline_mae,
        "improvement_pct": improvement,
        "beat_baseline": model_mae_valid < baseline_mae if not np.isnan(baseline_mae) else None,
        "best_iteration": model.best_iteration,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "top_features": top_features,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    log.info(
        "  %s — val MAE: %.3f | baseline MAE: %.3f | improvement: %.1f%% | best iter: %d",
        stat, model_mae_valid, baseline_mae, improvement, model.best_iteration,
    )
    return model, metadata


def save_model(model: XGBRegressor, metadata: dict, stat: str) -> Path:
    """Save model + metadata as a single joblib bundle."""
    _SAVED_DIR.mkdir(parents=True, exist_ok=True)
    path = _SAVED_DIR / f"{stat}_model.joblib"
    joblib.dump({"model": model, "metadata": metadata}, path)
    log.info("  Saved → %s", path)
    return path


def train_all(stats: list[str], features_path: Path | str = _DEFAULT_FEATURES) -> list[dict]:
    """Train and save models for all specified stats. Returns list of metadata dicts."""
    df = load_features(features_path)
    feature_cols = get_feature_columns(df)
    log.info("Using %d feature columns", len(feature_cols))

    results = []
    for stat in stats:
        if stat not in df.columns:
            log.warning("Stat '%s' not found in features — skipping", stat)
            continue
        model, metadata = train_model(stat, df, feature_cols)
        save_model(model, metadata, stat)
        results.append(metadata)

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'Stat':<12} {'Val MAE':>8} {'Baseline':>9} {'Improv%':>8} {'Beat?':>6} {'Iters':>6}")
    print("-" * 75)
    for m in results:
        beat = "YES" if m["beat_baseline"] else "NO"
        print(
            f"{m['stat']:<12} {m['val_mae_valid_subset']:>8.3f} {m['baseline_mae']:>9.3f} "
            f"{m['improvement_pct']:>7.1f}% {beat:>6} {m['best_iteration']:>6}"
        )
    print("=" * 75)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost models per stat")
    parser.add_argument(
        "--stats", nargs="+", default=None,
        help="Specific stats to train (default: P0 = points, rebounds, assists)",
    )
    parser.add_argument(
        "--all-stats", action="store_true",
        help="Train all P0 + P1 stats",
    )
    parser.add_argument(
        "--features", default=str(_DEFAULT_FEATURES),
        help=f"Path to features parquet (default: {_DEFAULT_FEATURES})",
    )
    args = parser.parse_args()

    if args.stats:
        stats = args.stats
    elif args.all_stats:
        stats = P0_STATS + P1_STATS
    else:
        stats = P0_STATS

    train_all(stats, features_path=args.features)


if __name__ == "__main__":
    main()
