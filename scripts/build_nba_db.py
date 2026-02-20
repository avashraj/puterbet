#!/usr/bin/env python3
"""
Build NBA SQLite database from ESPN public APIs.
Supports --test (short run), --retry-failed, and --season.
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy.exc import IntegrityError

from database import SessionLocal
from models import BoxScore, Game, Player, Team

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
MAX_RETRIES = 5
RATE_LIMIT_SLEEP = 0.5

SEASON_RANGES = {
    "2022-23": (date(2022, 10, 18), date(2023, 6, 12)),
    "2023-24": (date(2023, 10, 24), date(2024, 6, 17)),
    "2024-25": (date(2024, 10, 22), date(2025, 6, 22)),
    "2025-26": (date(2025, 10, 28), date.today()),
}

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
CHECKPOINT_PATH = LOG_DIR / "checkpoint.json"
FAILED_ROSTERS = LOG_DIR / "failed_rosters.log"
FAILED_GAMES = LOG_DIR / "failed_games.log"
FAILED_BOXSCORES = LOG_DIR / "failed_boxscores.log"
MAIN_LOG = LOG_DIR / "build_nba_db.log"

logger = logging.getLogger("build_nba_db")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt_verbose = logging.Formatter(
        "%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_short = logging.Formatter("%(levelname)-5s %(message)s")

    fh = RotatingFileHandler(MAIN_LOG, maxBytes=10 * 1024 * 1024, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_verbose)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_short)
    root.addHandler(ch)


def fetch_json(
    session: requests.Session, path: str, params: dict | None = None
) -> dict | None:
    url = f"{ESPN_BASE}/{path}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                time.sleep(RATE_LIMIT_SLEEP)
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = 2**attempt
                logger.warning(
                    "HTTP %s on %s, retrying in %ss (attempt %s/%s)",
                    resp.status_code,
                    url,
                    wait,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(wait)
                continue
            logger.error("HTTP %s on %s, skipping", resp.status_code, url)
            return None
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = 2**attempt
            logger.warning(
                "%s on %s, retrying in %ss (attempt %s/%s)",
                type(e).__name__,
                url,
                wait,
                attempt + 1,
                MAX_RETRIES,
            )
            time.sleep(wait)
    logger.error("Exhausted retries for %s", url)
    return None


def append_failed_id(log_path: Path, id_val: str) -> None:
    try:
        with open(log_path, "a") as f:
            f.write(id_val.strip() + "\n")
    except OSError as e:
        logger.warning("Could not append to %s: %s", log_path, e)


def read_failed_ids(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    with open(log_path) as f:
        return [line.strip() for line in f if line.strip()]


def remove_failed_id(log_path: Path, id_val: str) -> None:
    if not log_path.exists():
        return
    try:
        with open(log_path) as f:
            lines = [line for line in f if line.strip() != str(id_val).strip()]
        with open(log_path, "w") as f:
            f.writelines(lines)
    except OSError as e:
        logger.warning("Could not update %s: %s", log_path, e)


def _default_checkpoint() -> dict:
    return {
        "run_id": datetime.now().isoformat(timespec="seconds"),
        "phases_completed": [],
        "games_phase": {
            s: {"last_completed_date": None, "games_inserted": 0}
            for s in SEASON_RANGES
        },
        "boxscores_phase": {"games_completed": 0, "games_remaining": 0},
        "stats": {"api_calls": 0, "total_errors": 0, "elapsed_seconds": 0},
    }


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        return _default_checkpoint()
    try:
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load checkpoint: %s. Starting fresh.", e)
        return _default_checkpoint()


def save_checkpoint(data: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, CHECKPOINT_PATH)
    except OSError as e:
        logger.warning("Could not save checkpoint: %s", e)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def phase1_teams(session: requests.Session, db: SessionLocal, checkpoint: dict) -> bool:
    if "teams" in checkpoint.get("phases_completed", []):
        logger.info("[Phase 1] Teams already in checkpoint, skipping.")
        return True
    data = fetch_json(session, "teams")
    if not data:
        logger.error("[Phase 1] Failed to fetch teams. Aborting.")
        return False
    try:
        teams_data = (
            data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        )
    except (IndexError, KeyError, TypeError):
        logger.error("[Phase 1] Unexpected teams response structure.")
        return False
    count = 0
    for t in teams_data:
        team = t.get("team") or t
        tid = team.get("id")
        if tid is None:
            continue
        try:
            tid = int(tid)
        except (ValueError, TypeError):
            continue
        name = team.get("displayName") or team.get("name") or ""
        city = team.get("location") or ""
        abbr = team.get("abbreviation") or ""
        if not name or not abbr:
            continue
        db.merge(Team(id=tid, name=name, city=city, abbreviation=abbr))
        count += 1
    db.commit()
    checkpoint.setdefault("phases_completed", []).append("teams")
    save_checkpoint(checkpoint)
    logger.info("[Phase 1] Loaded %s teams.", count)
    return True


def phase2_players(
    session: requests.Session,
    db: SessionLocal,
    checkpoint: dict,
    *,
    team_ids: list[int] | None = None,
    retry_failed: bool = False,
) -> None:
    did_full_rosters = False
    if retry_failed:
        team_ids = [int(x) for x in read_failed_ids(FAILED_ROSTERS) if x.isdigit()]
        if not team_ids:
            logger.info("[Phase 2] No failed rosters to retry.")
            return
    elif "players" in checkpoint.get("phases_completed", []) and team_ids is None:
        logger.info("[Phase 2] Players already in checkpoint, skipping.")
        return
    else:
        if team_ids is None:
            team_ids = [r.id for r in db.query(Team.id).all()]
            did_full_rosters = True

    for tid in team_ids:
        data = fetch_json(session, f"teams/{tid}/roster")
        if not data:
            append_failed_id(FAILED_ROSTERS, str(tid))
            continue
        try:
            athletes = data.get("athletes", []) or data.get("roster", [])
        except (TypeError, AttributeError):
            athletes = []
        for a in athletes:
            ath = a.get("athlete") or a
            pid = ath.get("id")
            if pid is None:
                continue
            try:
                pid = int(pid)
            except (ValueError, TypeError):
                continue
            first = ath.get("firstName") or ath.get("fullName") or ""
            last = ath.get("lastName") or ""
            if not last and " " in first:
                parts = first.split(None, 1)
                first, last = parts[0], parts[1] if len(parts) > 1 else ""
            height = ath.get("height") or ath.get("displayHeight")
            if isinstance(height, dict):
                height = height.get("displayValue") or str(height.get("value", ""))
            weight = ath.get("weight") or ath.get("displayWeight")
            if isinstance(weight, dict):
                weight = weight.get("displayValue") or str(weight.get("value", ""))
            birth = ath.get("dateOfBirth") or ath.get("birthDate")
            db.merge(
                Player(
                    id=pid,
                    first_name=first or "Unknown",
                    last_name=last or "Unknown",
                    height=str(height)[:10] if height else None,
                    weight=str(weight)[:10] if weight else None,
                    birthdate=str(birth)[:20] if birth else None,
                    current_team_id=tid,
                )
            )
        db.commit()
        remove_failed_id(FAILED_ROSTERS, str(tid))
        logger.info("[Phase 2] Loaded roster for team %s (%s players).", tid, len(athletes))

    if not retry_failed and did_full_rosters:
        checkpoint.setdefault("phases_completed", []).append("players")
        save_checkpoint(checkpoint)


def _season_for_date(d: date) -> str:
    for season, (start, end) in SEASON_RANGES.items():
        if start <= d <= end:
            return season
    return ""


def _season_from_event(event: dict) -> str:
    season_info = event.get("season") or {}
    year = season_info.get("year")
    if year:
        return f"{year - 1}-{str(year)[2:]}"
    return ""


def _playoff_round_from_event(event: dict) -> str | None:
    season_info = event.get("season") or {}
    if int(season_info.get("type", 0)) != 3:
        return None
    name = (event.get("name") or "") + " " + (event.get("shortName") or "")
    if "final" in name.lower() or "finals" in name.lower():
        return "NBA Finals"
    if "conference final" in name.lower():
        return "Conference Finals"
    if "semifinal" in name.lower() or "semi-final" in name.lower():
        return "Conference Semifinals"
    if "first round" in name.lower() or "round 1" in name.lower():
        return "First Round"
    return "Playoffs"


def phase3_games(
    session: requests.Session,
    db: SessionLocal,
    checkpoint: dict,
    *,
    test_dates: list[date] | None = None,
    season_filter: str | None = None,
    retry_failed: bool = False,
) -> int:
    if retry_failed:
        dates_str = read_failed_ids(FAILED_GAMES)
        dates = []
        for d in dates_str:
            if len(d) == 8 and d.isdigit():
                try:
                    dates.append(
                        date(int(d[:4]), int(d[4:6]), int(d[6:8]))
                    )
                except ValueError:
                    pass
        if not dates:
            logger.info("[Phase 3] No failed game dates to retry.")
            return 0
        test_dates = sorted(dates)
        season_filter = None
    elif test_dates:
        pass
    else:
        seasons = [season_filter] if season_filter else list(SEASON_RANGES)
        test_dates = []
        for season in seasons:
            if season not in SEASON_RANGES:
                continue
            start, end = SEASON_RANGES[season]
            last_d = checkpoint.get("games_phase", {}).get(season, {}).get(
                "last_completed_date"
            )
            if last_d:
                try:
                    start = datetime.strptime(last_d, "%Y-%m-%d").date() + timedelta(
                        days=1
                    )
                except ValueError:
                    pass
            d = start
            while d <= end:
                test_dates.append(d)
                d += timedelta(days=1)

    total = 0
    games_phase = checkpoint.get("games_phase", {}).copy()
    for d in test_dates:
        date_str = d.strftime("%Y-%m-%d")
        ymd = d.strftime("%Y%m%d")
        data = fetch_json(session, "scoreboard", {"dates": ymd})
        if not data:
            append_failed_id(FAILED_GAMES, ymd)
            continue
        events = data.get("events") or []
        season_label = ""
        inserted = 0
        for ev in events:
            eid = ev.get("id")
            if not eid:
                continue
            try:
                eid = int(eid)
            except (ValueError, TypeError):
                continue
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors") or []
            if len(competitors) < 2:
                continue
            season_info = ev.get("season") or {}
            if int(season_info.get("type", 0)) == 1:
                continue
            if not season_label:
                season_label = _season_from_event(ev)
            home = away = None
            home_score = away_score = None
            for c in competitors:
                t = c.get("team") or {}
                tid = t.get("id")
                if tid is None:
                    continue
                try:
                    tid = int(tid)
                except (ValueError, TypeError):
                    continue
                if (c.get("homeAway") or "").lower() == "home":
                    home = tid
                    try:
                        home_score = int(c.get("score", 0) or 0)
                    except (ValueError, TypeError):
                        home_score = None
                else:
                    away = tid
                    try:
                        away_score = int(c.get("score", 0) or 0)
                    except (ValueError, TypeError):
                        away_score = None
            if home is None or away is None:
                continue
            if db.get(Game, eid) is not None:
                continue
            playoff_round = _playoff_round_from_event(ev)
            db.add(
                Game(
                    id=eid,
                    home_team_id=home,
                    away_team_id=away,
                    date=date_str,
                    season=season_label or "unknown",
                    playoff_round=playoff_round,
                    home_score=home_score,
                    away_score=away_score,
                )
            )
            inserted += 1
        db.commit()
        total += inserted
        season_key = season_label or _season_for_date(d)
        if season_key:
            games_phase[season_key] = games_phase.get(season_key) or {}
            games_phase[season_key]["last_completed_date"] = date_str
            prev = games_phase[season_key].get("games_inserted", 0)
            games_phase[season_key]["games_inserted"] = prev + inserted
        checkpoint["games_phase"] = games_phase
        save_checkpoint(checkpoint)
        logger.info("[Phase 3] %s: %s games (total this run: %s).", date_str, inserted, total)
    return total


def _parse_made_attempted(s: str) -> tuple[int | None, int | None]:
    if not s or "-" not in s:
        return None, None
    parts = s.split("-", 1)
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, TypeError, IndexError):
        return None, None


def _safe_int(val, default: int | None = None) -> int | None:
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def phase4_boxscores(
    session: requests.Session,
    db: SessionLocal,
    checkpoint: dict,
    *,
    game_ids: list[int] | None = None,
    retry_failed: bool = False,
) -> int:
    if retry_failed:
        game_ids = [
            int(x)
            for x in read_failed_ids(FAILED_BOXSCORES)
            if x.isdigit()
        ]
        if not game_ids:
            logger.info("[Phase 4] No failed box scores to retry.")
            return 0
    elif game_ids is None:
        subq = (
            db.query(Game.id)
            .outerjoin(BoxScore, Game.id == BoxScore.game_id)
            .filter(BoxScore.id.is_(None))
        )
        game_ids = [r[0] for r in subq.all()]

    completed = 0
    keys_order = [
        "minutes",
        "points",
        "fieldGoalsMade-fieldGoalsAttempted",
        "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
        "freeThrowsMade-freeThrowsAttempted",
        "rebounds",
        "assists",
        "turnovers",
        "steals",
        "blocks",
        "offensiveRebounds",
        "defensiveRebounds",
        "fouls",
        "plusMinus",
    ]
    for gid in game_ids:
        data = fetch_json(session, "summary", {"event": str(gid)})
        if not data:
            append_failed_id(FAILED_BOXSCORES, str(gid))
            continue
        box = data.get("boxscore") or {}
        players_list = box.get("players") or []
        try:
            for team_block in players_list:
                team_info = team_block.get("team") or {}
                team_id = team_info.get("id")
                if team_id is not None:
                    try:
                        team_id = int(team_id)
                    except (ValueError, TypeError):
                        continue
                else:
                    continue
                stats_block = team_block.get("statistics") or []
                if not stats_block:
                    continue
                keys = (stats_block[0].get("keys") or []) or keys_order
                athletes = stats_block[0].get("athletes") or []
                for ath_entry in athletes:
                    if ath_entry.get("didNotPlay"):
                        continue
                    athlete = ath_entry.get("athlete") or {}
                    pid = athlete.get("id")
                    if pid is None:
                        continue
                    try:
                        pid = int(pid)
                    except (ValueError, TypeError):
                        continue
                    if db.get(Player, pid) is None:
                        display = athlete.get("displayName") or athlete.get("fullName") or "Unknown"
                        parts = display.split(None, 1)
                        first = parts[0] if parts else "Unknown"
                        last = parts[1] if len(parts) > 1 else ""
                        db.add(
                            Player(
                                id=pid,
                                first_name=first,
                                last_name=last,
                                current_team_id=team_id,
                            )
                        )
                        db.flush()
                    stats_arr = ath_entry.get("stats") or []
                    if not stats_arr and ath_entry.get("didNotPlay"):
                        continue
                    starter = ath_entry.get("starter", False)
                    minutes = stats_arr[0] if len(stats_arr) > 0 else None
                    points = _safe_int(stats_arr[1]) if len(stats_arr) > 1 else None
                    fg_m, fg_a = _parse_made_attempted(stats_arr[2]) if len(stats_arr) > 2 else (None, None)
                    threes_m, threes_a = _parse_made_attempted(stats_arr[3]) if len(stats_arr) > 3 else (None, None)
                    ft_m, ft_a = _parse_made_attempted(stats_arr[4]) if len(stats_arr) > 4 else (None, None)
                    rebounds = _safe_int(stats_arr[5]) if len(stats_arr) > 5 else None
                    assists = _safe_int(stats_arr[6]) if len(stats_arr) > 6 else None
                    turnovers = _safe_int(stats_arr[7]) if len(stats_arr) > 7 else None
                    steals = _safe_int(stats_arr[8]) if len(stats_arr) > 8 else None
                    blocks = _safe_int(stats_arr[9]) if len(stats_arr) > 9 else None
                    pf = _safe_int(stats_arr[12]) if len(stats_arr) > 12 else None
                    plus_minus = _safe_int(stats_arr[13]) if len(stats_arr) > 13 else None
                    db.add(
                        BoxScore(
                            player_id=pid,
                            game_id=gid,
                            team_id=team_id,
                            points=points,
                            rebounds=rebounds,
                            assists=assists,
                            fgm=fg_m,
                            fga=fg_a,
                            three_pm=threes_m,
                            three_pa=threes_a,
                            ftm=ft_m,
                            fta=ft_a,
                            steals=steals,
                            blocks=blocks,
                            turnovers=turnovers,
                            pf=pf,
                            plus_minus=plus_minus,
                            minutes=str(minutes)[:10] if minutes else None,
                            starter=starter,
                        )
                    )
            db.commit()
            completed += 1
            remove_failed_id(FAILED_BOXSCORES, str(gid))
        except IntegrityError:
            db.rollback()
            logger.warning("Duplicate or constraint error for game %s, skipping.", gid)
            append_failed_id(FAILED_BOXSCORES, str(gid))
            continue
        except Exception as e:
            db.rollback()
            logger.exception("Box score parse error for game %s: %s", gid, e)
            append_failed_id(FAILED_BOXSCORES, str(gid))
            continue
        logger.info("[Phase 4] Box scores: game %s (%s completed).", gid, completed)

    checkpoint["boxscores_phase"] = checkpoint.get("boxscores_phase") or {}
    checkpoint["boxscores_phase"]["games_completed"] = (
        checkpoint["boxscores_phase"].get("games_completed", 0) + completed
    )
    save_checkpoint(checkpoint)
    return completed


def get_test_dates(season: str = "2025-26", num: int = 5) -> list[date]:
    if season not in SEASON_RANGES:
        season = "2025-26"
    start, end = SEASON_RANGES[season]
    # Pick last `num` dates that are in the past
    today = date.today()
    candidates = []
    d = end if end <= today else today
    while len(candidates) < num and d >= start:
        candidates.append(d)
        d -= timedelta(days=1)
    return candidates[:num]


def run_test_mode(
    session: requests.Session,
    db: SessionLocal,
    season: str | None,
) -> None:
    season = season or "2025-26"
    test_dates = get_test_dates(season, 5)
    if not test_dates:
        logger.warning("No test dates available for season %s.", season)
        return
    checkpoint = load_checkpoint()
    t0 = time.monotonic()
    phase1_teams(session, db, checkpoint)
    team_ids_in_games = set()
    for d in test_dates:
        ymd = d.strftime("%Y%m%d")
        data = fetch_json(session, "scoreboard", {"dates": ymd})
        if data:
            for ev in data.get("events") or []:
                for c in (ev.get("competitions") or [{}])[0].get("competitors") or []:
                    t = c.get("team") or {}
                    tid = t.get("id")
                    if tid is not None:
                        try:
                            team_ids_in_games.add(int(tid))
                        except (ValueError, TypeError):
                            pass
    phase2_players(session, db, checkpoint, team_ids=list(team_ids_in_games)[:5])
    games_inserted = phase3_games(
        session, db, checkpoint, test_dates=test_dates, season_filter=season
    )
    box_completed = phase4_boxscores(session, db, checkpoint, game_ids=None)
    elapsed = time.monotonic() - t0
    teams_count = db.query(Team).count()
    players_count = db.query(Player).count()
    games_count = db.query(Game).count()
    box_count = db.query(BoxScore).count()
    print("\n=== TEST MODE SUMMARY ===")
    print(f"Teams:      {teams_count} loaded")
    print(f"Players:    {players_count} loaded")
    print(f"Games:      {games_count} in DB ({games_inserted} inserted this run)")
    print(f"Box scores: {box_count} inserted")
    print(f"Duration:   {elapsed:.0f}s")
    print("Status:     PASS -- test phases completed\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NBA SQLite DB from ESPN APIs")
    parser.add_argument("--test", action="store_true", help="Run test mode (5 dates, ~1-2 min)")
    parser.add_argument("--retry-failed", action="store_true", help="Retry only failed IDs from logs")
    parser.add_argument("--season", type=str, help="Run only this season (e.g. 2022-23)")
    args = parser.parse_args()

    setup_logging()
    checkpoint = load_checkpoint()

    with requests.Session() as session:
        session.headers.setdefault("User-Agent", "puterbet/1.0")
        db = SessionLocal()
        try:
            if args.test:
                run_test_mode(session, db, args.season)
                return
            if args.retry_failed:
                phase2_players(session, db, checkpoint, retry_failed=True)
                phase3_games(session, db, checkpoint, retry_failed=True)
                phase4_boxscores(session, db, checkpoint, retry_failed=True)
                return
            if not phase1_teams(session, db, checkpoint):
                return
            phase2_players(session, db, checkpoint)
            phase3_games(session, db, checkpoint, season_filter=args.season)
            phase4_boxscores(session, db, checkpoint)
        finally:
            db.close()


if __name__ == "__main__":
    main()
