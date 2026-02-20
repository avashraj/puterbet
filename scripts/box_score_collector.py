import json


from nba_api.stats.endpoints import BoxScoreTraditionalV3

def read_all_game_ids(path):
    with open(path, "r") as f:
        data = json.load(f)

    return data




# id = "0022300182"
# score = BoxScoreTraditionalV3(game_id=id)
# df = score.player_stats.get_data_frame()
# print(df.columns)
#
#
path = "gameIds.json"
d = read_all_game_ids(path)
print(d)
