"""features/build_features.py

Orchestrator that builds the full feature matrix from the SQLite database and
writes it to a Parquet file.

Usage:
    python -m features.build_features [--db PATH] [--output PATH] [--no-validate]

Output Parquet schema (~200 columns):
    Identity:       player_id, game_id, team_id, date, season
    Targets:        points, rebounds, assists, three_pm, steals, blocks,
                    turnovers, minutes_played
    Context flags:  is_home, is_playoff, is_starter, dnp
    Schedule:       days_rest, is_back_to_back, games_last_7,
                    player_season_game_num, starter_status_changed, starts_last_5
    Rolling:        {stat}_roll_{mean|median|std}_{3|5|10|20},
                    {stat}_trend_10, {stat}_season_mean  (12 stats each)
    Home/away:      {stat}_roll_mean_10_{home|away}  (4 stats)
    Opponent:       opp_{pts|reb|ast|3pm|stl|blk}_allowed_roll10,
                    opp_pace_roll10, opp_def_rating_roll10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from features.opponent import compute_opponent_features
from features.player_rolling import compute_player_rolling_features
from features.schedule import compute_schedule_features, preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).parents[1] / "nba.db"
_DEFAULT_OUTPUT = Path(__file__).parents[1] / "data" / "features.parquet"

_LOAD_SQL = """
    SELECT
        bs.player_id,
        bs.game_id,
        bs.team_id,
        bs.minutes,
        bs.starter,
        bs.points,
        bs.rebounds,
        bs.assists,
        bs.fgm,
        bs.fga,
        bs.three_pm,
        bs.three_pa,
        bs.ftm,
        bs.fta,
        bs.steals,
        bs.blocks,
        bs.turnovers,
        bs.pf,
        bs.plus_minus,
        g.date,
        g.season,
        g.playoff_round,
        g.home_team_id,
        g.away_team_id
    FROM box_scores bs
    JOIN games g ON bs.game_id = g.id
    ORDER BY g.date ASC, g.id ASC, bs.player_id ASC
"""

_GAMES_SQL = """
    SELECT id AS game_id, home_team_id, away_team_id, date, season
    FROM games
"""


def load_raw_data(db_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw box scores and games from SQLite.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (player_df, games_df) where player_df has one row per player-game
        and games_df has one row per game.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    log.info("Loading box scores from %s ...", db_path)
    player_df = pd.read_sql(_LOAD_SQL, engine)
    games_df = pd.read_sql(_GAMES_SQL, engine)
    log.info(
        "Loaded %s player-game rows, %s games",
        f"{len(player_df):,}",
        f"{len(games_df):,}",
    )
    return player_df, games_df


def validate_no_leakage(
    features_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    n_rolling_samples: int = 100,
    n_opponent_samples: int = 50,
) -> None:
    """Run three leakage checks and raise ValueError on any failure.

    Check 1: Every player's first game must have NaN for points_roll_mean_3
             (no prior data to roll over after shift).
    Check 2: Spot-check rolling mean values against manual recomputation from
             raw data using strict date < game_date filtering.
    Check 3: opp_pts_allowed_roll10 must exclude the current game's result.

    Parameters
    ----------
    features_df : pd.DataFrame
        Full feature matrix output from build_feature_matrix().
    raw_df : pd.DataFrame
        Raw preprocessed DataFrame (before rolling features) for recomputation.
    n_rolling_samples : int
        Number of random rows to spot-check for rolling leakage.
    n_opponent_samples : int
        Number of random rows to spot-check for opponent feature leakage.
    """
    log.info("Running leakage validation ...")

    # --- Check 1: First-game NaN ---
    # .nth(0) returns the literal first row per group (including NaNs).
    # .first() would skip NaNs and return the first non-null value — wrong here.
    first_games = (
        features_df.sort_values(["player_id", "date", "game_id"])
        .groupby("player_id")
        .nth(0)
        .reset_index()
    )
    check_col = "points_roll_mean_3"
    if check_col in first_games.columns:
        violations = first_games[first_games[check_col].notna()]
        if len(violations) > 0:
            raise ValueError(
                f"Leakage check 1 failed: {len(violations)} players have non-NaN "
                f"{check_col} on their first game. "
                f"Example player_id={violations['player_id'].iloc[0]}"
            )
        log.info("Check 1 passed: first-game NaN check (%d players)", len(first_games))

    # --- Check 2: Spot-check rolling values ---
    if check_col not in features_df.columns:
        log.warning("points_roll_mean_10 not found; skipping rolling spot-check")
    else:
        # Prepare raw data with date_dt for filtering
        raw_with_dt = raw_df.copy()
        if "date_dt" not in raw_with_dt.columns:
            raw_with_dt["date_dt"] = pd.to_datetime(raw_with_dt["date"])

        # Sample rows that have enough history (game_num >= 10) to be meaningful
        eligible = features_df[features_df["player_season_game_num"] >= 5]
        if len(eligible) == 0:
            eligible = features_df
        sample = eligible.sample(
            n=min(n_rolling_samples, len(eligible)), random_state=42
        )

        for _, row in sample.iterrows():
            pid = int(row["player_id"])
            game_date = pd.Timestamp(row["date"])

            # Manually compute points_roll_mean_3 from raw data
            prior = raw_with_dt[
                (raw_with_dt["player_id"] == pid)
                & (raw_with_dt["date_dt"] < game_date)
                & (raw_with_dt["points"].notna())  # exclude DNP rows
            ].sort_values("date_dt")

            if len(prior) == 0:
                expected = np.nan
            else:
                expected = float(prior["points"].tail(3).mean())

            actual = row[check_col]

            both_nan = np.isnan(expected) if expected != expected else False
            if both_nan:
                if pd.notna(actual):
                    raise ValueError(
                        f"Leakage check 2 failed: player {pid}, game_id {row['game_id']}: "
                        f"expected NaN but got {actual:.3f}"
                    )
                continue

            if not np.isclose(expected, actual, atol=0.01, equal_nan=True):
                raise ValueError(
                    f"Leakage check 2 failed: player {pid}, game_id {row['game_id']}: "
                    f"expected {expected:.3f}, got {actual:.4f}"
                )

        log.info("Check 2 passed: rolling spot-check (%d samples)", len(sample))

    # --- Check 3: Opponent feature leakage ---
    # opp_pts_allowed_roll10 = rolling avg of points scored BY opp_tid's opponents
    # against them (= points opp_tid GAVE UP), not points opp_tid scored.
    opp_col = "opp_pts_allowed_roll10"
    if opp_col not in features_df.columns:
        log.warning("%s not found; skipping opponent leakage check", opp_col)
    else:
        raw_with_dt = raw_df.copy()
        if "date_dt" not in raw_with_dt.columns:
            raw_with_dt["date_dt"] = pd.to_datetime(raw_with_dt["date"])

        eligible_opp = features_df[features_df[opp_col].notna()]
        sample_opp = eligible_opp.sample(
            n=min(n_opponent_samples, len(eligible_opp)), random_state=99
        )

        for _, row in sample_opp.iterrows():
            game_date = pd.Timestamp(row["date"])
            opp_tid = (
                int(row["away_team_id"])
                if int(row["is_home"]) == 1
                else int(row["home_team_id"])
            )

            # Find games where opp_tid played BEFORE game_date (sorted by date)
            opp_game_dates = (
                raw_with_dt[raw_with_dt["team_id"] == opp_tid][["game_id", "date_dt"]]
                .drop_duplicates("game_id")
                .sort_values("date_dt")
            )
            opp_prior_game_dates = opp_game_dates[
                opp_game_dates["date_dt"] < game_date
            ]
            if len(opp_prior_game_dates) == 0:
                continue

            # Take the last 10 games (same window as the feature)
            last_10_ids = set(opp_prior_game_dates["game_id"].tail(10).values)

            # Points scored BY opp_tid's opponents in those games = what opp_tid ALLOWED
            # i.e., players whose team_id != opp_tid in those games
            pts_against_opp = (
                raw_with_dt[
                    raw_with_dt["game_id"].isin(last_10_ids)
                    & (raw_with_dt["team_id"] != opp_tid)
                ]
                .groupby("game_id")["points"]
                .sum()
            )
            if len(pts_against_opp) == 0:
                continue

            expected_opp = float(pts_against_opp.mean())
            actual_opp = float(row[opp_col])

            if not np.isclose(expected_opp, actual_opp, atol=0.5, equal_nan=True):
                raise ValueError(
                    f"Leakage check 3 failed: player {row['player_id']}, "
                    f"game_id {row['game_id']}, opp_team {opp_tid}: "
                    f"expected pts_allowed avg {expected_opp:.2f}, got {actual_opp:.2f}"
                )

        log.info("Check 3 passed: opponent spot-check (%d samples)", len(sample_opp))

    log.info("All leakage checks passed.")


def build_feature_matrix(
    db_path: str | Path = _DEFAULT_DB,
    output_path: str | Path = _DEFAULT_OUTPUT,
    validate_leakage: bool = True,
) -> pd.DataFrame:
    """Build the full feature matrix from the SQLite database.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database.
    output_path : str | Path
        Path to write the output Parquet file. Parent directory is created if
        it doesn't exist.
    validate_leakage : bool
        If True, run leakage validation checks and raise ValueError on failure.

    Returns
    -------
    pd.DataFrame
        Full feature matrix, also written to output_path as Parquet.
    """
    db_path = Path(db_path)
    output_path = Path(output_path)

    # Step 1: Load raw data
    player_df, games_df = load_raw_data(db_path)

    # Step 2: Preprocess (parse minutes, derive flags, null DNP stats)
    log.info("Preprocessing ...")
    player_df = preprocess(player_df)

    # Keep a copy of the preprocessed raw data before rolling for leakage validation
    raw_for_validation = player_df.copy()

    # Step 3: Schedule features (sorts by player_id, date_dt, game_id)
    log.info("Computing schedule features ...")
    player_df = compute_schedule_features(player_df)

    # Step 4: Player rolling features
    log.info("Computing player rolling features (may take ~30s) ...")
    player_df = compute_player_rolling_features(player_df)

    # Step 5: Opponent features
    log.info("Computing opponent defensive features ...")
    player_df = compute_opponent_features(player_df, games_df)

    log.info(
        "Feature matrix shape: %d rows × %d columns", player_df.shape[0], player_df.shape[1]
    )

    # Step 6: Leakage validation
    if validate_leakage:
        validate_no_leakage(player_df, raw_for_validation)

    # Step 7: Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    player_df.to_parquet(output_path, index=False)
    log.info("Feature matrix written to %s", output_path)

    return player_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build NBA player props feature matrix"
    )
    parser.add_argument(
        "--db",
        default=str(_DEFAULT_DB),
        help=f"Path to SQLite database (default: {_DEFAULT_DB})",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help=f"Output Parquet path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip leakage validation checks",
    )
    args = parser.parse_args()

    df = build_feature_matrix(
        db_path=args.db,
        output_path=args.output,
        validate_leakage=not args.no_validate,
    )

    print(f"\nDone. Shape: {df.shape}")
    print(f"\nColumn groups:")
    identity = ["player_id", "game_id", "team_id", "date", "season"]
    targets = ["points", "rebounds", "assists", "three_pm", "steals", "blocks", "turnovers", "minutes_played"]
    schedule = ["days_rest", "is_back_to_back", "games_last_7", "player_season_game_num"]
    roll_cols = [c for c in df.columns if "roll_mean" in c or "roll_std" in c]
    opp_cols = [c for c in df.columns if c.startswith("opp_")]
    print(f"  Identity:  {len(identity)} columns")
    print(f"  Targets:   {len(targets)} columns")
    print(f"  Schedule:  {len(schedule)} columns")
    print(f"  Rolling:   {len(roll_cols)} columns")
    print(f"  Opponent:  {len(opp_cols)} columns")
    print(f"\nOverall NaN rate: {df.isnull().mean().mean():.3f}")
    print(f"\nSample (first 3 rows, select columns):")
    sample_cols = identity + targets[:3] + ["points_roll_mean_5", "opp_pts_allowed_roll10"]
    print(df[sample_cols].head(3).to_string())
