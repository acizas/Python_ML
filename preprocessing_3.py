import numpy
import csv
import pandas

def load_data():
    return pandas.read_csv('nba_players_stats_19_20_per_game.csv')

selected_features=['3P','FGA','FT']

def load_clean_normal_data():
    data = pandas.read_csv('nba_players_stats_19_20_per_game.csv')[['Player']+selected_features]
    for stat in selected_features:
        data[stat] = data[stat]/data[stat].max()
    return data
