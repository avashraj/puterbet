import json
from datetime import datetime
from nba_api.stats.endpoints import scheduleleaguev2

def main():
    # seasons = ["2023", "2024", "2025"]
    seasons = ["2025"]
    gameIds = {}
    today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for season in seasons:
        response: dict = scheduleleaguev2.ScheduleLeagueV2(season=season).get_dict()

        leagueSchedule: dict | None = response.get("leagueSchedule", None)
        if not leagueSchedule:
            print(f"LOG THAT {season} did not work for some reason could not find leagueSchedule")

        gameDates: list | None = leagueSchedule.get("gameDates", None) #type: ignore
        if not gameDates:
            print(f"LOG THAT {season} did not work for some reason; no gameDates")

        for gameDate in gameDates: #type: ignore
            day = gameDate.get("gameDate", None) #type: ignore
            if not day:
                print(f"LOG THAT {day} could not be found")

            date = datetime.strptime(day, "%m/%d/%Y %H:%M:%S")
            if date >= today_midnight:
                break

            games = gameDate.get("games", None) #type: ignore
            if not games:
                print("fartbutt log some shit")

            for game in games:
                gameId = game.get("gameId", None)
                if not gameId:
                    print("log some shit")

                gameType = game.get("gameLabel", "")
                if gameType == "Preseason" or gameId in gameIds:
                    break
                    # continue

                gameIds[gameId] = gameType

    with open("gameIds.json", "w") as f:
        json.dump(gameIds, f)


if __name__ == "__main__":
    main()
