import requests
import sqlite3
import json
import time
from datetime import datetime

# Configuration
DB_NAME = "nba_data.db"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

# NBA Season Date Ranges (Regular Seasons)
SEASONS = [
    {"season": "2022-23", "start": "20221018", "end": "20230409"},
    {"season": "2023-24", "start": "20231024", "end": "20240414"},
    {"season": "2024-25", "start": "20241022", "end": "20250413"},
    {"season": "2025-26", "start": "20251021", "end": "20260219"} # Current date (Adjusted to today)
]

def setup_db():
    conn = sqlite3.connect(DB_NAME)
    curr = conn.cursor()
    curr.execute('''
        CREATE TABLE IF NOT EXISTS box_scores (
            game_id TEXT PRIMARY KEY,
            season TEXT,
            game_date TEXT,
            game_name TEXT,
            raw_json TEXT
        )
    ''')
    conn.commit()
    return conn

def get_game_ids(start_date, end_date):
    """Fetches all game IDs between two dates."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={start_date}-{end_date}&limit=1000"
    try:
        res = requests.get(url, headers=HEADERS)
        events = res.json().get('events', [])
        return events
    except Exception as e:
        print(f"Error fetching IDs: {e}")
        return []

def get_box_score(game_id):
    """Fetches full box score JSON for a game."""
    url = f"https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
    try:
        res = requests.get(url, headers=HEADERS)
        return res.text # Return as string for SQLite
    except Exception as e:
        print(f"Error fetching game {game_id}: {e}")
        return None

def main():
    db = setup_db()
    cursor = db.cursor()

    for season in SEASONS:
        print(f"--- Processing Season {season['season']} ---")
        games = get_game_ids(season['start'], season['end'])

        for game in games:
            g_id = game['id']
            g_date = game['date']
            g_name = game['name']

            # Skip if already in DB
            cursor.execute("SELECT 1 FROM box_scores WHERE game_id=?", (g_id,))
            if cursor.fetchone():
                continue

            print(f"Fetching: {g_name} ({g_id})")
            raw_data = get_box_score(g_id)

            if raw_data:
                cursor.execute(
                    "INSERT INTO box_scores (game_id, season, game_date, game_name, raw_json) VALUES (?, ?, ?, ?, ?)",
                    (g_id, season['season'], g_date, g_name, raw_data)
                )
                db.commit()

            # Rate limiting sleep (Polite scraping)
            time.sleep(0.5)

    db.close()
    print("All data successfully stored!")

if __name__ == "__main__":
    main()
