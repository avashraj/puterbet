"""models_ml/evaluate.py

Evaluate trained XGBoost models against baseline and compute betting-relevant metrics.

Usage:
    uv run python -m models_ml.evaluate                     # Val set, P0 stats
    uv run python -m models_ml.evaluate --split test        # Test set
    uv run python -m models_ml.evaluate --all-stats         # P0 + P1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models_ml.train import (
    DEFAULT_PARAMS,
    P0_STATS,
    P1_STATS,
    TEST_SEASON,
    VAL_SEASON,
    load_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_FEATURES = Path(__file__).parents[1] / "data" / "features.parquet"
_SAVED_DIR = Path(__file__).parent / "saved"


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error, ignoring NaN pairs."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid])))


def compute_directional_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, lines: np.ndarray
) -> float:
    """Fraction of predictions where model and actual agree on over/under vs the line.

    Only evaluates rows where both prediction and line are non-NaN and
    the prediction differs from the line.
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(lines))
    if valid.sum() == 0:
        return float("nan")

    y_t, y_p, l = y_true[valid], y_pred[valid], lines[valid]

    # Skip rows where prediction equals the line exactly (no edge, no bet)
    has_edge = y_p != l
    if has_edge.sum() == 0:
        return float("nan")

    y_t, y_p, l = y_t[has_edge], y_p[has_edge], l[has_edge]

    pred_over = y_p > l
    actual_over = y_t > l
    return float(np.mean(pred_over == actual_over))


def compute_calibration(
    y_true: np.ndarray, y_pred: np.ndarray, lines: np.ndarray, n_bins: int = 5
) -> list[dict]:
    """Bin predictions by edge magnitude, compute actual over-rate per bin.

    Returns a list of dicts with bin info for display.
    """
    valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(lines))
    if valid.sum() == 0:
        return []

    y_t, y_p, l = y_true[valid], y_pred[valid], lines[valid]
    edge = y_p - l  # positive = model predicts over

    # Bin by edge magnitude
    bins = np.percentile(edge, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    bin_idx = np.digitize(edge, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    results = []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        actual_over_rate = float(np.mean(y_t[mask] > l[mask]))
        mean_edge = float(np.mean(edge[mask]))
        results.append({
            "bin": i,
            "n": int(mask.sum()),
            "mean_edge": mean_edge,
            "actual_over_rate": actual_over_rate,
            "pred_direction": "over" if mean_edge > 0 else "under",
        })
    return results


def evaluate_model(
    stat: str,
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str = "val",
) -> dict:
    """Full evaluation of a model on a given split.

    Returns dict with MAE, baseline MAE, improvement, directional accuracy, calibration.
    """
    season = VAL_SEASON if split == "val" else TEST_SEASON
    split_df = df[df["season"] == season].copy()

    if len(split_df) == 0:
        log.warning("No data for split '%s' (season %s)", split, season)
        return {"stat": stat, "split": split, "error": "no data"}

    X = split_df[feature_cols].values.astype(np.float32)
    y_true = split_df[stat].values.astype(np.float32)
    y_pred = model.predict(X).astype(np.float32)

    # Baseline: rolling mean 10
    baseline_col = f"{stat}_roll_mean_10"
    lines = split_df[baseline_col].values.astype(np.float32) if baseline_col in split_df.columns else np.full(len(y_true), np.nan)

    # MAE
    model_mae = compute_mae(y_true, y_pred)
    baseline_mae = compute_mae(y_true, lines)
    improvement = (baseline_mae - model_mae) / baseline_mae * 100 if baseline_mae > 0 else float("nan")

    # Directional accuracy (using rolling mean as proxy line)
    dir_acc = compute_directional_accuracy(y_true, y_pred, lines)

    # Calibration
    calibration = compute_calibration(y_true, y_pred, lines)

    return {
        "stat": stat,
        "split": split,
        "season": season,
        "n_rows": len(split_df),
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "improvement_pct": improvement,
        "beat_baseline": model_mae < baseline_mae,
        "directional_accuracy": dir_acc,
        "calibration": calibration,
    }


def load_model(stat: str) -> tuple:
    """Load a saved model bundle. Returns (model, metadata)."""
    path = _SAVED_DIR / f"{stat}_model.joblib"
    bundle = joblib.load(path)
    return bundle["model"], bundle["metadata"]


def evaluate_all(
    stats: list[str],
    split: str = "val",
    features_path: Path | str = _DEFAULT_FEATURES,
) -> list[dict]:
    """Evaluate all specified stat models. Returns list of result dicts."""
    df = load_features(features_path)
    results = []

    for stat in stats:
        path = _SAVED_DIR / f"{stat}_model.joblib"
        if not path.exists():
            log.warning("No saved model for '%s' — skipping", stat)
            continue

        model, metadata = load_model(stat)
        feature_cols = metadata["feature_cols"]
        result = evaluate_model(stat, model, df, feature_cols, split)
        results.append(result)

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"Evaluation on {split} set")
    print(f"{'=' * 80}")
    print(f"{'Stat':<12} {'N':>7} {'Model MAE':>10} {'Baseline':>9} {'Improv%':>8} {'Beat?':>6} {'Dir Acc':>8}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['stat']:<12} ERROR: {r['error']}")
            continue
        beat = "YES" if r["beat_baseline"] else "NO"
        print(
            f"{r['stat']:<12} {r['n_rows']:>7} {r['model_mae']:>10.3f} "
            f"{r['baseline_mae']:>9.3f} {r['improvement_pct']:>7.1f}% "
            f"{beat:>6} {r['directional_accuracy']:>8.3f}"
        )
    print("=" * 80)

    # Print calibration for each stat
    for r in results:
        if "error" in r or not r.get("calibration"):
            continue
        print(f"\nCalibration — {r['stat']}:")
        print(f"  {'Bin':>4} {'N':>6} {'Mean Edge':>10} {'Actual Over%':>13}")
        for b in r["calibration"]:
            print(f"  {b['bin']:>4} {b['n']:>6} {b['mean_edge']:>+10.2f} {b['actual_over_rate']:>12.1%}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained XGBoost models")
    parser.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Which split to evaluate on (default: val)",
    )
    parser.add_argument(
        "--stats", nargs="+", default=None,
        help="Specific stats to evaluate (default: P0)",
    )
    parser.add_argument(
        "--all-stats", action="store_true",
        help="Evaluate all P0 + P1 stats",
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

    evaluate_all(stats, split=args.split, features_path=args.features)


if __name__ == "__main__":
    main()
