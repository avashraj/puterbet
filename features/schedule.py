"""features/schedule.py

Preprocessing and schedule/context features for the NBA player props feature engine.

This module is responsible for:
  - Parsing raw ESPN data (minutes strings, nullable booleans)
  - Deriving basic context flags (is_home, is_playoff, dnp)
  - Computing schedule features (days_rest, is_back_to_back, games_last_7)
  - Nulling out production stats for DNP rows so rolling windows ignore them

Run standalone to validate:
    python -m features.schedule
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Production stat columns — nulled for DNP rows before rolling computation
STAT_COLS: list[str] = [
    "points",
    "rebounds",
    "assists",
    "three_pm",
    "steals",
    "blocks",
    "turnovers",
    "fgm",
    "fga",
    "ftm",
    "fta",
]


def parse_minutes(s: str | None) -> float:
    """Parse ESPN minutes string to float.

    Handles:
      - None / NULL  → 0.0
      - '--'         → 0.0  (player logged but never entered the game)
      - integer str  → float(s)
    """
    if s is None or s == "--":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw box score columns and add basic derived flags.

    Must be called before any feature computation. Returns a new DataFrame.

    Adds:
        minutes_played  float64  parsed minutes
        dnp             int8     1 if player never entered (minutes_played == 0)
        is_home         int8     1 if player's team is the home team
        is_playoff      int8     1 if playoff_round is not null
        is_starter      int8     1 if player started
        date_dt         datetime64[ns]  parsed game date

    Also nulls out STAT_COLS for DNP rows so rolling windows skip them.
    """
    df = df.copy()
    df["minutes_played"] = df["minutes"].apply(parse_minutes)
    df["dnp"] = (df["minutes_played"] == 0.0).astype("int8")
    df["is_home"] = (df["team_id"] == df["home_team_id"]).astype("int8")
    df["is_playoff"] = df["playoff_round"].notna().astype("int8")
    df["is_starter"] = df["starter"].fillna(False).astype("int8")
    df["date_dt"] = pd.to_datetime(df["date"])

    # Null production stats for DNP rows — rolling windows treat NaN as missing,
    # so DNP appearances don't pollute a player's rolling averages.
    dnp_mask = df["dnp"] == 1
    for col in STAT_COLS:
        df.loc[dnp_mask, col] = np.nan

    return df


def _count_games_in_window(dates_sorted: np.ndarray, window_days: int = 7) -> np.ndarray:
    """For each game at index i, count prior games within window_days calendar days.

    Uses binary search (O(n log n) per player) rather than a full O(n^2) scan.

    Parameters
    ----------
    dates_sorted : np.ndarray
        datetime64 array sorted ascending. Must be for a single player.
    window_days : int
        Lookback window in calendar days (exclusive of window boundary).

    Returns
    -------
    np.ndarray of int64, same length as dates_sorted.
        counts[i] = number of games in (dates_sorted[i] - window_days, dates_sorted[i]).
    """
    n = len(dates_sorted)
    counts = np.zeros(n, dtype=np.int64)
    if n == 0:
        return counts
    window = np.timedelta64(window_days, "D")
    for i in range(1, n):
        cutoff = dates_sorted[i] - window
        # Find leftmost index where date > cutoff (strict inequality)
        left = np.searchsorted(dates_sorted[:i], cutoff, side="right")
        counts[i] = i - left
    return counts


def compute_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute schedule and context features for each player-game row.

    Must be called on a preprocessed DataFrame (after preprocess()). Sorts by
    (player_id, date_dt, game_id) internally and returns the sorted DataFrame.

    Adds columns:
        days_rest               float64  days since player's prior game (NaN = first game)
        is_back_to_back         int8     1 if days_rest == 0
        games_last_7            int64    prior games in past 7 calendar days
        player_season_game_num  int64    player's Nth game in the current season (1-indexed)
        starter_status_changed  int8     1 if starter status differs from prior game
        starts_last_5           float64  starts in prior 5 games

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: player_id, date_dt, game_id, season, is_starter.

    Returns
    -------
    pd.DataFrame
        Sorted by (player_id, date_dt, game_id) with new feature columns appended.
    """
    df = df.sort_values(["player_id", "date_dt", "game_id"]).reset_index(drop=True)

    # --- Days rest and back-to-back ---
    # days_rest = number of full rest days between games (0 = back-to-back = consecutive days)
    # elapsed_days(B2B) = 1 (Oct 18 → Oct 19), so rest_days = elapsed - 1 = 0
    prev_date = df.groupby("player_id", sort=False)["date_dt"].shift(1)
    elapsed = (df["date_dt"] - prev_date).dt.days
    df["days_rest"] = (elapsed - 1).astype("float64")  # NaN stays NaN for first game
    df["is_back_to_back"] = (df["days_rest"] == 0).astype("int8")

    # --- Games in prior 7 calendar days ---
    def _apply_games_last_7(grp: pd.DataFrame) -> pd.Series:
        dates = grp["date_dt"].values
        counts = _count_games_in_window(dates, window_days=7)
        return pd.Series(counts, index=grp.index)

    df["games_last_7"] = df.groupby("player_id", group_keys=False).apply(
        _apply_games_last_7
    )

    # --- Season game number (resets per season) ---
    df["player_season_game_num"] = (
        df.groupby(["player_id", "season"]).cumcount() + 1
    ).astype("int64")

    # --- Starter context ---
    # shift(1) gives the previous game's starter status; NaN for first game
    shifted_starter = df.groupby("player_id", sort=False)["is_starter"].shift(1)
    df["starter_status_changed"] = (df["is_starter"] != shifted_starter).astype("int8")

    # starts_last_5: number of starts in the 5 games immediately before this game
    df["starts_last_5"] = shifted_starter.groupby(
        df["player_id"], sort=False
    ).transform(lambda x: x.rolling(5, min_periods=1).sum())

    return df


if __name__ == "__main__":
    from pathlib import Path

    from sqlalchemy import create_engine

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
    print(f"Loaded {len(raw):,} rows, {raw.shape[1]} columns")

    df = preprocess(raw)
    print(f"After preprocess: {df.shape[1]} columns")
    print(f"  DNP rows: {df['dnp'].sum():,}")
    print(f"  is_home: {df['is_home'].mean():.3f} (expect ~0.50)")

    df = compute_schedule_features(df)

    new_cols = [
        "days_rest",
        "is_back_to_back",
        "games_last_7",
        "player_season_game_num",
        "starter_status_changed",
        "starts_last_5",
    ]
    print(f"\nSchedule feature summary:")
    print(df[new_cols].describe().round(3))
    print(f"\nNaN rates:")
    print(df[new_cols].isnull().mean().round(4))
    print(f"\nSample rows:")
    print(df[["player_id", "date", "days_rest", "is_back_to_back", "games_last_7"]].head(10).to_string())
