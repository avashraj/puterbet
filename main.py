import json
from nba_api.stats.endpoints import playercareerstats

# Nikola Jokić
def poop():
    career = playercareerstats.PlayerCareerStats(player_id='203999')
    d = career.get_dict()
    print(d)


def main():
    path = "gameIds.json"
    with open(path, "r") as f:
        d = json.load(f)

    print(f"{len(d)} Game Ids")
    regular_season = 0

    res = {}
    for k, v in d.items():
        if v == "":
            regular_season += 1
        else:
            res[v] = res.get(v, 0) + 1

    print(f"Regular Season games: {regular_season}")
    for k, v in res.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    poop()
