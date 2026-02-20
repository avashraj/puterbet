"""features/opponent.py

Opponent defensive context features for the NBA player props feature engine.

For each player-game row, computes rolling averages of the opponent team's
defensive stats (points/rebounds/etc. allowed) over their prior games. All
features are walk-forward safe: opponent's defensive average for game N uses
only the opponent's games played before date N.

Run standalone to validate:
    python -m features.opponent
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Stats to compute opponent-allowed rolling averages for
ALLOWED_STAT_COLS: list[str] = [
    "points",
    "rebounds",
    "assists",
    "three_pm",
    "steals",
    "blocks",
]

DEFAULT_WINDOW: int = 10


def _build_team_game_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate box scores to team-game level.

    Parameters
    ----------
    player_df : pd.DataFrame
        Preprocessed player box score data. NaN in stat columns = DNP row
        (already set by preprocess()). We use sum(min_count=1) to handle
        games where all players have NaN for a stat.

    Returns
    -------
    pd.DataFrame
        One row per (team_id, game_id) with summed team stats:
        pts_scored, reb_scored, ast_scored, three_pm_scored, stl_scored,
        blk_scored, fga_total, fta_total, tov_total, pace_approx.
    """
    agg = (
        player_df.groupby(["team_id", "game_id"])
        .agg(
            pts_scored=("points", "sum"),
            reb_scored=("rebounds", "sum"),
            ast_scored=("assists", "sum"),
            three_pm_scored=("three_pm", "sum"),
            stl_scored=("steals", "sum"),
            blk_scored=("blocks", "sum"),
            fga_total=("fga", "sum"),
            fta_total=("fta", "sum"),
            tov_total=("turnovers", "sum"),
        )
        .reset_index()
    )
    # Pace approximation: FGA + 0.44*FTA + TOV (no offensive rebound data available)
    agg["pace_approx"] = (
        agg["fga_total"] + 0.44 * agg["fta_total"] + agg["tov_total"]
    )
    return agg


def _build_allowed_stats(
    team_game: pd.DataFrame, games_df: pd.DataFrame
) -> pd.DataFrame:
    """Build per-team-per-game allowed stats.

    "What team X allowed" = "what team X's opponent produced against them".

    For each game, expands to two rows: one for each team facing the other.

    Parameters
    ----------
    team_game : pd.DataFrame
        Output of _build_team_game_stats().
    games_df : pd.DataFrame
        One row per game with game_id, home_team_id, away_team_id, date.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, game_id, game_date_dt,
                 allowed_points, allowed_rebounds, allowed_assists,
                 allowed_three_pm, allowed_steals, allowed_blocks,
                 team_pace, def_rating_approx.
        team_id = the defending team, allowed_* = what opponent produced.
    """
    # Build matchup table: two rows per game (home perspective, away perspective)
    home = games_df[["game_id", "home_team_id", "away_team_id"]].rename(
        columns={"home_team_id": "team_id", "away_team_id": "opp_team_id"}
    )
    away = games_df[["game_id", "away_team_id", "home_team_id"]].rename(
        columns={"away_team_id": "team_id", "home_team_id": "opp_team_id"}
    )
    matchup = pd.concat([home, away], ignore_index=True)

    # What the opponent produced = what team_id allowed
    opp_stats = team_game[
        [
            "team_id",
            "game_id",
            "pts_scored",
            "reb_scored",
            "ast_scored",
            "three_pm_scored",
            "stl_scored",
            "blk_scored",
        ]
    ].rename(
        columns={
            "team_id": "opp_team_id",
            "pts_scored": "allowed_points",
            "reb_scored": "allowed_rebounds",
            "ast_scored": "allowed_assists",
            "three_pm_scored": "allowed_three_pm",
            "stl_scored": "allowed_steals",
            "blk_scored": "allowed_blocks",
        }
    )

    allowed = matchup.merge(opp_stats, on=["game_id", "opp_team_id"], how="left")

    # Join the defending team's own pace (for defensive rating computation)
    team_pace = team_game[["team_id", "game_id", "pace_approx"]].rename(
        columns={"pace_approx": "team_pace"}
    )
    allowed = allowed.merge(team_pace, on=["team_id", "game_id"], how="left")

    # Per-game defensive rating: points allowed per ~100 possessions
    allowed["def_rating_approx"] = (
        allowed["allowed_points"] / allowed["team_pace"].replace(0.0, np.nan) * 100.0
    )

    # Join game date for walk-forward sorting
    allowed = allowed.merge(
        games_df[["game_id", "date"]].assign(
            game_date_dt=lambda x: pd.to_datetime(x["date"])
        ),
        on="game_id",
        how="left",
    )

    return allowed


def compute_opponent_features(
    player_df: pd.DataFrame,
    games_df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Compute opponent defensive context features for each player-game row.

    Parameters
    ----------
    player_df : pd.DataFrame
        Preprocessed player box score data with columns: player_id, game_id,
        team_id, is_home, home_team_id, away_team_id, date_dt, and all stat cols.
    games_df : pd.DataFrame
        One row per game: game_id, home_team_id, away_team_id, date.
    window : int
        Rolling window for opponent defensive averages. Default 10.

    Returns
    -------
    pd.DataFrame
        Same rows as player_df (same index) with additional columns:
            opp_pts_allowed_roll{window}
            opp_reb_allowed_roll{window}
            opp_ast_allowed_roll{window}
            opp_3pm_allowed_roll{window}
            opp_stl_allowed_roll{window}
            opp_blk_allowed_roll{window}
            opp_pace_roll{window}
            opp_def_rating_roll{window}
    """
    # Step 1: Team-game totals
    team_game = _build_team_game_stats(player_df)

    # Step 2: Allowed stats per team per game
    allowed = _build_allowed_stats(team_game, games_df)

    # Step 3: Walk-forward rolling per team
    # Sort by (team_id, game_date_dt, game_id) then shift(1) before rolling
    allowed = allowed.sort_values(
        ["team_id", "game_date_dt", "game_id"]
    ).reset_index(drop=True)

    roll_cols_map: dict[str, str] = {
        "allowed_points": f"opp_pts_allowed_roll{window}",
        "allowed_rebounds": f"opp_reb_allowed_roll{window}",
        "allowed_assists": f"opp_ast_allowed_roll{window}",
        "allowed_three_pm": f"opp_3pm_allowed_roll{window}",
        "allowed_steals": f"opp_stl_allowed_roll{window}",
        "allowed_blocks": f"opp_blk_allowed_roll{window}",
        "team_pace": f"opp_pace_roll{window}",
        "def_rating_approx": f"opp_def_rating_roll{window}",
    }

    for src_col, dst_col in roll_cols_map.items():
        shifted = allowed.groupby("team_id", sort=False)[src_col].shift(1)
        allowed[dst_col] = shifted.groupby(
            allowed["team_id"], sort=False
        ).transform(lambda x, w=window: x.rolling(w, min_periods=1).mean())

    # Step 4: Derive opponent team_id for each player-game
    # Player's opponent = the OTHER team in the game
    player_df = player_df.copy()
    player_df["opp_team_id"] = np.where(
        player_df["is_home"] == 1,
        player_df["away_team_id"],
        player_df["home_team_id"],
    )

    # Step 5: Merge rolling opponent stats back to player rows
    # Join on: player's opp_team_id == allowed.team_id AND game_id
    opp_feature_cols = list(roll_cols_map.values())
    allowed_slim = allowed[["team_id", "game_id"] + opp_feature_cols].rename(
        columns={"team_id": "opp_team_id"}
    )

    result = player_df.merge(allowed_slim, on=["opp_team_id", "game_id"], how="left")

    # Drop the helper column, preserve original index order
    result = result.drop(columns=["opp_team_id"])

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
    games_sql = """
        SELECT id as game_id, home_team_id, away_team_id, date, season
        FROM games
    """
    raw = pd.read_sql(SQL, engine)
    games_df = pd.read_sql(games_sql, engine)
    print(f"Loaded {len(raw):,} player rows, {len(games_df):,} games")

    df = preprocess(raw)
    df = compute_schedule_features(df)
    df = compute_opponent_features(df, games_df)

    opp_cols = [c for c in df.columns if c.startswith("opp_")]
    print(f"\nOpponent feature columns ({len(opp_cols)}): {opp_cols}")
    print(f"\nNaN rates:")
    print(df[opp_cols].isnull().mean().round(4))
    print(f"\nStats summary:")
    print(df[opp_cols].describe().round(2))
    print(f"\nSample (5 rows):")
    print(
        df[["player_id", "date", "team_id"] + opp_cols]
        .dropna(subset=[opp_cols[0]])
        .head(5)
        .to_string()
    )
