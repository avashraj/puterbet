# CLAUDE.md — NBA Player Props Prediction System

## Project Overview

**Name:** PuterBet (working title)
**Goal:** Build a player prop prediction engine for NBA DFS-style platforms (PrizePicks, Underdog, Betr, etc.) that generates daily pick recommendations with confidence scores.

**Core thesis:** Player prop lines on DFS platforms are softer than traditional sportsbook game lines. By modeling player stat outputs with matchup-aware features derived from historical box scores, we can identify +EV (positive expected value) plays where our predicted distribution diverges from the posted line.

---

## Current State

### Data Available
- **SQLite database** built from ESPN public APIs via `build_nba_db.py`
- **Seasons covered:** 2022-23, 2023-24, 2024-25, 2025-26 (through today)
- **Tables:**
  - `teams` — 30 NBA teams (id, name, city, abbreviation)
  - `players` — All rostered players (id, first_name, last_name, height, weight, birthdate, current_team_id)
  - `games` — All regular season + playoff games (id, home_team_id, away_team_id, date, season, playoff_round, home_score, away_score)
  - `box_scores` — Per-player per-game stat lines (player_id, game_id, team_id, points, rebounds, assists, fgm, fga, three_pm, three_pa, ftm, fta, steals, blocks, turnovers, pf, plus_minus, minutes, starter)

### Data NOT Yet Available (Future Iterations)
- Historical prop lines / odds (critical for backtesting profitability, not just accuracy)
- Real-time injury reports and lineup confirmations
- Advanced play-by-play / tracking data
- Usage rate / pace data
- Referee assignments

---

## Architecture

### MVP (Phase 1) — Local CLI Tool

```
┌──────────────────────────────────────────────────────┐
│                    SQLite DB                          │
│  (teams, players, games, box_scores)                 │
└──────────────┬───────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │ Feature Engine  │  Computes per-player features
       │  (Python)       │  from historical box scores
       └───────┬─────────┘
               │
       ┌───────▼────────┐
       │   Model Layer   │  One regression model per
       │  (XGBoost)      │  stat category
       └───────┬─────────┘
               │
       ┌───────▼────────┐
       │  Prediction &   │  Compare predictions to lines,
       │  Ranking Layer  │  rank by edge size + confidence
       └───────┬─────────┘
               │
       ┌───────▼────────┐
       │  Daily Output   │  CLI report or simple web page
       │  (picks.json /  │  showing today's best plays
       │   terminal)     │
       └────────────────┘
```

### Future (Phase 2+) — Hosted Service

```
┌────────────┐   ┌──────────────┐   ┌──────────────────┐
│  Data       │   │  Prop Line   │   │  Injury/Lineup   │
│  Ingestion  │──▶│  Scraper     │──▶│  Monitor         │
│  (daily)    │   │  (PrizePicks │   │  (Twitter/ESPN)  │
│             │   │   Underdog)  │   │                  │
└──────┬──────┘   └──────┬───────┘   └───────┬──────────┘
       │                 │                    │
       ▼                 ▼                    ▼
┌──────────────────────────────────────────────────────┐
│                   PostgreSQL                          │
└──────────────────────┬───────────────────────────────┘
                       │
               ┌───────▼────────┐
               │ Feature Engine  │
               │ + Model Layer   │
               └───────┬─────────┘
                       │
               ┌───────▼────────┐
               │   API Server    │  FastAPI / Flask
               │   (REST + WS)  │
               └───────┬─────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     ┌─────────┐ ┌──────────┐ ┌──────────┐
     │  Web    │ │  Mobile  │ │  Discord │
     │  App    │ │  App     │ │  Bot     │
     └─────────┘ └──────────┘ └──────────┘
```

---

## Prop Categories to Model

Start with the highest-volume PrizePicks/Underdog markets:

| Priority | Stat         | Column(s) in DB            | Notes                              |
|----------|-------------|----------------------------|------------------------------------|
| P0       | Points      | `points`                   | Most liquid market                 |
| P0       | Rebounds    | `rebounds`                 | High volume                        |
| P0       | Assists     | `assists`                  | High volume                        |
| P1       | 3-Pointers  | `three_pm`                 | Popular on PrizePicks              |
| P1       | Steals      | `steals`                   | Lower volume, higher variance      |
| P1       | Blocks      | `blocks`                   | Lower volume, higher variance      |
| P2       | PRA combo   | `points + rebounds + assists` | Common combo market             |
| P2       | PA combo    | `points + assists`         | Common combo market                |
| P2       | PR combo    | `points + rebounds`        | Common combo market                |

---

## Feature Engineering

This is where most of the edge will come from. All features are computed **per player, per game** using only data available *before* that game (no leakage).

### Player Rolling Stats
- Rolling mean of stat over last N games (N = 3, 5, 10, 20, season)
- Rolling median (more robust to blowout outliers)
- Rolling standard deviation (captures consistency)
- Trend: slope of linear fit over last 10 games (is the player heating up or cooling down?)
- Minutes rolling average (this is the single most predictive feature — if a player's minutes are trending up, everything else follows)

### Home/Away Splits
- Rolling averages split by home vs. away games
- Binary `is_home` feature for the prediction game

### Rest & Schedule
- Days since last game (0 = back-to-back)
- `is_back_to_back` flag
- Games played in last 7 days (fatigue proxy)

### Opponent Defensive Context
- Opponent's defensive rating (points allowed per 100 possessions, approximated from box scores)
- Opponent's allowed stats by position/role: e.g., how many points do opposing players in this minutes-tier score against this team? (Since we don't have positional data, use minutes played as a proxy for role)
- Opponent's pace (possessions per game, estimated from box scores: `pace ≈ FGA + 0.44*FTA - ORB + TOV`)
- Opponent's allowed stat rolling averages over last 10 games (recent form matters more than season average)

### Starter Context
- `is_starter` (from box score data)
- Has starter status changed recently? (player moving in/out of starting lineup)

### Season Context
- Game number in season (early season is noisier)
- Is playoff game (different intensity, rotations tighten)

---

## Modeling Strategy

### Approach: Per-Stat Regression
- Train one **XGBoost regressor** per stat category (points, rebounds, assists, etc.)
- Input: feature vector for (player, game)
- Output: predicted stat value
- Also capture prediction intervals using quantile regression or conformal prediction to get over/under probabilities, not just point estimates

### Training & Validation
- **Train set:** 2022-23 and 2023-24 seasons
- **Validation set:** 2024-25 season (tune hyperparameters here)
- **Test set:** 2025-26 season (final evaluation, simulate live betting)
- **Important:** Features must be computed using only past data relative to each game (walk-forward approach). Never use future game data to compute features for a past game.

### Baseline to Beat
- **Naive baseline:** Player's last-10-game rolling average for each stat
- If the model can't beat this baseline meaningfully, the features or model need work before going further

### Evaluation Metrics
- **MAE** (Mean Absolute Error) — how far off are predictions on average?
- **Directional accuracy** — when prediction > line, did the player actually go over? (This is what matters for betting)
- **Calibration** — if model says 60% chance of going over, does it actually go over ~60% of the time?
- **Simulated profit** — assuming flat bet sizing and standard -110 juice (or PrizePicks payout structure), would following the model's picks have been profitable?

---

## Daily Workflow (MVP)

```
1. UPDATE DATA
   └─ Run build_nba_db.py to pull yesterday's games + box scores

2. GET TODAY'S GAMES
   └─ Query ESPN scoreboard API for today's matchups
   └─ Get list of players likely to play (starters + rotation players)

3. COMPUTE FEATURES
   └─ For each player in today's games, compute all features from historical data

4. GENERATE PREDICTIONS
   └─ Run each stat model to get predicted values + confidence intervals
   └─ Output: {player, stat, predicted_value, p_over, p_under}

5. COMPARE TO LINES (manual for MVP)
   └─ User inputs prop lines from PrizePicks/Underdog
   └─ System computes edge: (model_probability - implied_probability)

6. RANK PLAYS
   └─ Sort by edge size * confidence
   └─ Filter: minimum minutes threshold (>20 min avg), minimum games played (>10)
   └─ Output top 5-10 plays with reasoning

7. TRACK RESULTS
   └─ After games complete, log prediction vs actual vs line
   └─ Track cumulative accuracy and simulated P&L
```

---

## Project Structure

```
puterbet/
├── CLAUDE.md                  # This file
├── README.md                  # User-facing docs
├── requirements.txt           # Python dependencies
├── database.py                # SQLAlchemy engine + session setup
├── models.py                  # ORM models (Team, Player, Game, BoxScore)
├── build_nba_db.py            # ESPN data ingestion script
│
├── features/
│   ├── __init__.py
│   ├── player_rolling.py      # Rolling averages, medians, trends
│   ├── opponent.py            # Opponent defensive features
│   ├── schedule.py            # Rest days, B2B, fatigue
│   └── build_features.py      # Orchestrator: builds full feature matrix
│
├── models_ml/
│   ├── __init__.py
│   ├── train.py               # Train XGBoost models per stat
│   ├── predict.py             # Generate predictions for today's games
│   ├── evaluate.py            # Backtest evaluation + metrics
│   └── saved/                 # Serialized model files (.joblib)
│
├── picks/
│   ├── __init__.py
│   ├── generate_picks.py      # Main daily picks pipeline
│   ├── compare_lines.py       # Edge calculation vs prop lines
│   └── tracker.py             # Results tracking + P&L logging
│
├── data/
│   └── nba.db                 # SQLite database
│
├── logs/
│   ├── build_nba_db.log
│   ├── checkpoint.json
│   └── picks_history/         # Daily pick logs for tracking
│
└── scripts/
    ├── daily_run.sh            # Cron-friendly: update data → generate picks
    └── backtest.py             # Run full historical backtest
```

---

## Tech Stack

| Component        | MVP Choice       | Future Scale         |
|-----------------|------------------|----------------------|
| Language         | Python 3.11+     | Python + TypeScript  |
| Database         | SQLite           | PostgreSQL           |
| ORM              | SQLAlchemy       | SQLAlchemy           |
| ML Framework     | XGBoost          | XGBoost + LightGBM  |
| Data Processing  | Pandas           | Pandas / Polars      |
| API Server       | —                | FastAPI              |
| Task Scheduler   | cron / manual    | Prefect / Airflow    |
| Frontend         | Terminal / JSON   | React / Next.js      |
| Hosting          | Local machine    | Railway / Fly.io / AWS |
| Monitoring       | Log files        | Grafana / Datadog    |

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| No historical prop lines data | Can't backtest profitability, only accuracy | Start collecting lines now (even manually). Use model accuracy vs rolling-average baseline as proxy. |
| Injuries not in model | Best pick becomes worst pick if star teammate is out | Manual injury check before placing bets (MVP). Automate via ESPN injury API or Twitter later. |
| Overfitting to historical data | Model looks great in backtest, fails live | Strict walk-forward validation. Keep features simple. Monitor live performance weekly. |
| ESPN API changes/breaks | Data pipeline stops working | Retry logic already built in. Add monitoring/alerting. Consider backup data sources. |
| Platform line movement | Line moves before you can bet | Speed matters less for DFS platforms (lines are stickier). Focus on early-morning picks. |
| Low edge magnitude | Model is accurate but edge isn't big enough to overcome variance | Need volume — bet many small edges. Track Kelly criterion for sizing. |

---

## MVP Milestones

- [ ] **M1: Feature Engine** — Build feature computation module, validate no data leakage, output feature matrix for full dataset
- [ ] **M2: Baseline Model** — Train XGBoost for points/rebounds/assists, compare to naive rolling average baseline on 2024-25 holdout
- [ ] **M3: Prediction Pipeline** — End-to-end: today's games → feature computation → predictions → ranked output in terminal
- [ ] **M4: Manual Backtesting** — Collect PrizePicks lines for 1-2 weeks, compare model predictions, track results
- [ ] **M5: Results Tracker** — Automated logging of predictions vs actuals vs lines, cumulative P&L dashboard
- [ ] **M6: Line Scraping** — Automate prop line collection from PrizePicks/Underdog
- [ ] **M7: Web Interface** — Simple dashboard showing today's picks, historical performance

---

## Coding Conventions

- Python 3.11+ with type hints throughout
- Use `ruff` for linting and formatting
- SQLAlchemy ORM for all database access (already established)
- Pandas DataFrames for feature computation and model I/O
- All features must be computed using only temporally prior data — assert this in tests
- Model training and prediction must be reproducible (set random seeds, log hyperparameters)
- Every prediction logged with timestamp, model version, features used

---

## Commands Reference

```bash
# Update database with latest games
python build_nba_db.py --season 2025-26

# Retry any failed API calls
python build_nba_db.py --retry-failed

# Test mode (last 5 dates only)
python build_nba_db.py --test

# Build feature matrix (future)
python -m features.build_features --output data/features.parquet

# Train models (future)
python -m models_ml.train --stat points --train-seasons 2022-23,2023-24 --val-season 2024-25

# Generate today's picks (future)
python -m picks.generate_picks --date today

# Run backtest (future)
python scripts/backtest.py --season 2025-26 --stat points
```

---

## Future Enhancements (Post-MVP Backlog)

1. **Parlay builder** — Combine picks with correlation awareness (e.g., two players on same team are correlated, same-game parlays need joint modeling)
2. **Live line monitoring** — Track line movement and alert when model edge exceeds threshold
3. **Player prop line scraper** — Automated collection from PrizePicks/Underdog APIs or web scraping
4. **Injury impact model** — Quantify how Player X being out affects Player Y's stat projections (usage redistribution)
5. **Minute projection model** — Separate model to predict minutes, use as upstream input to stat models
6. **Pace-adjusted projections** — Normalize stats by pace; a player in a fast-paced game has more opportunities
7. **Subscription service** — Paid tier with Discord bot or web app for daily picks
8. **Multi-sport expansion** — NFL, MLB player props using similar architecture
