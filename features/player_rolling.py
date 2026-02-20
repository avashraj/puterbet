"""features/player_rolling.py

Per-player rolling stat features for the NBA player props feature engine.

All rolling features are computed walk-forward safe: features for game N use only
data from games 0..N-1 for that player, enforced via shift(1) before every rolling
window.

Run standalone to validate:
    python -m features.player_rolling
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Stats to compute rolling features for
STAT_COLS: list[str] = [
    "points",
    "rebounds",
    "assists",
    "three_pm",
    "steals",
    "blocks",
    "turnovers",
    "minutes_played",
    "fgm",
    "fga",
    "ftm",
    "fta",
]

# Stats to compute home/away splits for (subset to avoid column explosion)
HOME_AWAY_STAT_COLS: list[str] = [
    "points",
    "rebounds",
    "assists",
    "minutes_played",
]

DEFAULT_WINDOWS: list[int] = [3, 5, 10, 20]
TREND_WINDOW: int = 10


def _slope_raw(arr: np.ndarray) -> float:
    """Slope of a linear fit over valid (non-NaN) values in arr.

    Used as a rolling.apply callback with raw=True.
    Returns NaN if fewer than 2 valid observations.
    """
    valid = arr[~np.isnan(arr)]
    if len(valid) < 2:
        return np.nan
    x = np.arange(len(valid), dtype=np.float64)
    try:
        return float(np.polyfit(x, valid, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def compute_player_rolling_features(
    df: pd.DataFrame,
    stat_cols: list[str] | None = None,
    windows: list[int] | None = None,
    trend_window: int = TREND_WINDOW,
) -> pd.DataFrame:
    """Compute all player rolling features.

    Must be called on a preprocessed DataFrame (after preprocess() in schedule.py)
    so that 'minutes_played', 'is_home', 'dnp', and 'date_dt' columns exist, and
    STAT_COLS are already NaN for DNP rows.

    Sorts by (player_id, date_dt, game_id) internally.

    Parameters
    ----------
    df : pd.DataFrame
        Contains player_id, date_dt, game_id, season, is_home, is_starter, dnp,
        and all stat columns.
    stat_cols : list[str] | None
        Stats to build rolling features for. Defaults to STAT_COLS.
    windows : list[int] | None
        Rolling window sizes. Defaults to [3, 5, 10, 20].
    trend_window : int
        Window size for linear trend slope. Defaults to 10.

    Returns
    -------
    pd.DataFrame
        Same rows as input sorted by (player_id, date_dt, game_id), with
        additional rolling feature columns appended. Players' first games have
        NaN for all rolling features (no prior data).
    """
    stat_cols = stat_cols or STAT_COLS
    windows = windows or DEFAULT_WINDOWS

    df = df.sort_values(["player_id", "date_dt", "game_id"]).reset_index(drop=True)

    # Compute rolling features only over played games (dnp == 0).
    # If we include DNP rows (where stats are NaN), they occupy window slots but
    # contribute nothing to the mean — so a player with 2 DNPs between played games
    # would have a rolling(3) window that only sees 1 real game instead of 3.
    # Computing on the played subset ensures rolling(N) = last N actual appearances.
    played = df[df["dnp"] == 0].copy()
    played_pid = played["player_id"]
    played_season = played["season"]

    # Collect new columns keyed by (player_id, game_id) for merging back.
    new_cols: dict[str, pd.Series] = {}

    for stat in stat_cols:
        # shift(1) within player groups on the played-only subset
        shifted = played.groupby("player_id", sort=False)[stat].shift(1)

        # Rolling mean, median, std
        for w in windows:
            grp = shifted.groupby(played_pid, sort=False)
            new_cols[f"{stat}_roll_mean_{w}"] = grp.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).mean()
            )
            new_cols[f"{stat}_roll_median_{w}"] = grp.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).median()
            )
            new_cols[f"{stat}_roll_std_{w}"] = grp.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).std()
            )

        # Linear trend (slope) over last trend_window games
        new_cols[f"{stat}_trend_{trend_window}"] = shifted.groupby(
            played_pid, sort=False
        ).transform(
            lambda x, tw=trend_window: x.rolling(tw, min_periods=2).apply(
                _slope_raw, raw=True
            )
        )

        # Season-to-date expanding mean (resets at season boundaries)
        # shift within (player_id, season) so game 1 of each season starts NaN
        shifted_season = played.groupby(
            ["player_id", "season"], sort=False
        )[stat].shift(1)
        new_cols[f"{stat}_season_mean"] = shifted_season.groupby(
            [played_pid, played_season], sort=False
        ).transform(lambda x: x.expanding(min_periods=1).mean())

    # Home / away rolling means (within played games only)
    for stat in HOME_AWAY_STAT_COLS:
        for loc, flag in [("home", 1), ("away", 0)]:
            loc_stat = played[stat].where(played["is_home"] == flag, other=np.nan)
            shifted_loc = loc_stat.groupby(played_pid, sort=False).shift(1)
            new_cols[f"{stat}_roll_mean_10_{loc}"] = shifted_loc.groupby(
                played_pid, sort=False
            ).transform(lambda x: x.rolling(10, min_periods=1).mean())

    # Build feature DataFrame on played rows, then merge back to all rows
    feat_df = pd.DataFrame(new_cols, index=played.index)
    feat_df["player_id"] = played["player_id"].values
    feat_df["game_id"] = played["game_id"].values

    # Left join: DNP rows get NaN for all rolling features (correct — they're
    # excluded from model training anyway via the dnp flag).
    result = df.merge(feat_df, on=["player_id", "game_id"], how="left")

    return result


if __name__ == "__main__":
    from pathlib import Path

    from sqlalchemy import create_engine

    from features.schedule import compute_schedule_features, preprocess

    DB_PATH = Path(__file__).parents[1] / "nba.db"
    engine = create_engine(f"sqlite:///{DB_PATH}")

    SQL = """
        SELECT
            bs.player_id, bs.game_id, bs.team_id,
            bs.minutes, bs.starter,
            bs.points, bs.rebounds, bs.assists, bs.three_pm, bs.steals,
            bs.blocks, bs.turnovers, bs.fgm, bs.fga, bs.ftm, bs.fta,
            g.date, g.season, g.playoff_round, g.home_team_id, g.away_team_id
        FROM box_scores bs
        JOIN games g ON bs.game_id = g.id
        ORDER BY g.date ASC, g.id ASC, bs.player_id ASC
    """
    raw = pd.read_sql(SQL, engine)
    print(f"Loaded {len(raw):,} rows")

    df = preprocess(raw)
    df = compute_schedule_features(df)
    print("Running player rolling features (this may take ~30s)...")
    df = compute_player_rolling_features(df)

    roll_cols = [c for c in df.columns if "roll_mean" in c or "roll_std" in c or "trend" in c]
    print(f"\nTotal rolling feature columns: {len(roll_cols)}")
    print(f"DataFrame shape: {df.shape}")

    sample_cols = [
        "player_id", "date", "points",
        "points_roll_mean_5", "points_roll_mean_10", "points_season_mean",
        "points_trend_10", "minutes_played_roll_mean_10",
    ]
    print(f"\nNaN rates for key columns:")
    print(df[sample_cols[2:]].isnull().mean().round(4))
    print(f"\nSample (first player's first 5 games):")
    first_pid = df["player_id"].iloc[0]
    print(df[df["player_id"] == first_pid][sample_cols].head(5).to_string())
