# analysis.py
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def find_closest_players_per_stat(dfs, player_name, position_weights):
    closest_players_per_stat = {}
    combined_distances = {}

    first_df = next(iter(dfs.values()))
    player_position = first_df[first_df['player'] == player_name]['pos'].values[0]

    for stat_name, df in dfs.items():
        if player_name not in df['player'].values:
            continue

        pca_columns = [col for col in df.columns if col.startswith('pca')]
        player_row = df[df['player'] == player_name][pca_columns].iloc[[0]]  # Ensure a single row
        distances = cdist(player_row, df[pca_columns], metric='euclidean').flatten()
        df['distance'] = distances

        weight = position_weights.get(stat_name, 1.0)
        df['weighted_distance'] = df['distance'] * weight

        def position_bonus(other_position):
            if player_position == other_position:
                return -0.1
            elif (player_position == "D" and other_position == "M") or \
                 (player_position == "M" and other_position in ["D", "F"]) or \
                 (player_position == "F" and other_position == "M"):
                return -0.05
            else:
                return 0.1

        df['position_bonus'] = df['pos'].apply(position_bonus)
        df['final_distance'] = df['weighted_distance'] + df['position_bonus']

        min_distance = df['final_distance'].min()
        max_distance = df['final_distance'].max()
        df['similarity'] = 100 * (1 - ((df['final_distance'] - min_distance) / (max_distance - min_distance)))

        closest_players = df[df['player'] != player_name].sort_values(by='final_distance').head(10)
        closest_players_per_stat[stat_name] = closest_players[['player', 'pos', 'age', 'squad', 'final_distance', 'similarity']]

        for _, row in closest_players.iterrows():
            player = row['player']
            combined_distances[player] = combined_distances.get(player, 0) + row['final_distance']

    combined_df = pd.DataFrame(list(combined_distances.items()), columns=['player', 'total_weighted_distance'])
    combined_df = combined_df.merge(first_df[['player', 'pos', 'age', 'squad']], on='player', how='left')

    min_comb = combined_df['total_weighted_distance'].min()
    max_comb = combined_df['total_weighted_distance'].max()
    combined_df['similarity'] = 100 * (1 - ((combined_df['total_weighted_distance'] - min_comb) / (max_comb - min_comb)))
    combined_df['total_weighted_distance'] = combined_df['total_weighted_distance'].clip(0)
    combined_df['similarity'] = (combined_df['similarity'] * (1 - abs(combined_df['total_weighted_distance'])))
    combined_df['similarity'] = combined_df['similarity'].apply(lambda x: f"{x:.2f}%")

    return closest_players_per_stat, combined_df.sort_values(by='total_weighted_distance').head(10)

def compute_stat_group_means(dfs, player_name):
    stat_group_means = {}

    for stat_name, df in dfs.items():
        if player_name not in df['player'].values:
            continue

        numerical_cols = df.select_dtypes(include=np.number).columns
        if numerical_cols.empty:
            continue

        player_row = df[df['player'] == player_name][numerical_cols]
        if player_row.empty:
            continue

        group_means = df[numerical_cols].mean(axis=1)
        player_mean = player_row.mean(axis=1).values[0]
        percentile_rank = (group_means < player_mean).mean() * 100

        stat_group_means[stat_name] = percentile_rank

    return stat_group_means

def get_position_weights(selected_player, dfs):
    first_df = next(iter(dfs.values()))
    player_position = first_df[first_df['player'] == selected_player]['pos'].values[0]

    weights = {
        "standard_stats": 1.0,
        "shooting": 1.0,
        "passing": 1.0,
        "pass_types": 1.0,
        "goal_and_shot_creation": 1.0,
        "defensive_actions": 1.0,
        "possession": 1.0,
        "playing_time": 1.0,
        "miscellaneous_stats": 1.0,
    }

    if "F" in player_position:
        weights.update({
            "shooting": 1.5,
            "goal_and_shot_creation": 1.5,
            "passing": 1.2,
            "defensive_actions": 0.5,
        })
    elif "M" in player_position:
        weights.update({
            "passing": 1.5,
            "possession": 1.5,
            "goal_and_shot_creation": 1.3,
            "shooting": 1.0,
            "defensive_actions": 1.0,
        })
    elif "D" in player_position:
        weights.update({
            "defensive_actions": 1.5,
            "possession": 1.3,
            "playing_time": 1.2,
            "shooting": 0.5,
            "goal_and_shot_creation": 0.5,
        })

    return weights
