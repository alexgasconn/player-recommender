# fbref_analysis.py

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from functools import reduce
import warnings
from scipy.spatial.distance import cdist
import random

warnings.filterwarnings("ignore")

urls = {
    "standard_stats": "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats",
    "goalkeeping": "https://fbref.com/en/comps/Big5/keepers/players/Big-5-European-Leagues-Stats",
    "advanced_goalkeeping": "https://fbref.com/en/comps/Big5/keepersadv/players/Big-5-European-Leagues-Stats",
    "shooting": "https://fbref.com/en/comps/Big5/shooting/players/Big-5-European-Leagues-Stats",
    "passing": "https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats",
    "pass_types": "https://fbref.com/en/comps/Big5/passing_types/players/Big-5-European-Leagues-Stats",
    "gca": "https://fbref.com/en/comps/Big5/gca/players/Big-5-European-Leagues-Stats",
    "defense": "https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Stats",
    "possession": "https://fbref.com/en/comps/Big5/possession/players/Big-5-European-Leagues-Stats",
    "playing_time": "https://fbref.com/en/comps/Big5/playingtime/players/Big-5-European-Leagues-Stats",
    "misc": "https://fbref.com/en/comps/Big5/misc/players/Big-5-European-Leagues-Stats"
}

stat_keys = ["misc", "shooting", "passing", "defense", "pass_types", "gca", "standard_stats"]

# Load and process each dataframe
def load_and_process(url_key):
    print(f"Loading and processing: {url_key}")
    df = pd.read_html(urls[url_key])[0]
    df.columns = df.columns.droplevel()
    df.columns = df.columns.str.strip().str.lower()

    categorical_cols = ['player', 'nation', 'pos', 'squad', 'comp', 'age', 'born', 'matches']
    for col in df.columns:
        if col not in categorical_cols:
            # Ensure the column is a valid Series before converting
            if isinstance(df[col], pd.Series):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"WARNING: Skipping column '{col}' as it is not a valid Series")

    df.fillna(df.median(numeric_only=True), inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    print(f"Finished processing: {url_key} (rows: {df.shape[0]}, cols: {df.shape[1]})")
    return df

# Apply PCA + KMeans to each stat dataframe
def apply_pca_kmeans(df, name):
    print(f"Applying PCA and KMeans for: {name}")
    try:
        print("  Running PCA...")
        df_numerical = df.select_dtypes(include=np.number)
        if df_numerical.isnull().values.any():
            print(f"  WARNING: NaN values detected in numerical data for {name} BEFORE PCA")
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df_numerical)
        print("  PCA completed")

        df[f'pca1_{name}'] = df_pca[:, 0]
        df[f'pca2_{name}'] = df_pca[:, 1]

        print("  Checking for NaNs in PCA result before KMeans...")
        if np.isnan(df_pca).any():
            print(f"  ERROR: NaNs present in PCA-transformed data for {name}")
            raise ValueError(f"NaNs detected in PCA output for {name}")

        print("  Running KMeans...")
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        kmeans.fit(df_pca)
        print("  KMeans completed")

        df[f'cluster_{name}'] = kmeans.labels_
        print(f"Completed clustering for: {name}")

    except Exception as e:
        print(f"  ERROR during PCA/KMeans for {name}: {e}")
        raise

    return df

def find_closest_players(df_all, player_name):
    """
    Find the top 10 closest players to the selected player based on Euclidean distances in PCA space.
    
    Args:
        df_all (pd.DataFrame): The merged DataFrame containing PCA components for all stats.
        player_name (str): The name of the player to compare against.
    
    Returns:
        pd.DataFrame: A DataFrame containing the top 10 closest players for each PCA space.
    """
    if player_name not in df_all['player'].values:
        print(f"Player '{player_name}' not found in the dataset.")
        return None

    # Filter rows with PCA columns only
    pca_columns = [col for col in df_all.columns if col.startswith('pca')]
    if not pca_columns:
        print("No PCA columns found in the dataset.")
        return None

    # Get the PCA values for the selected player
    player_row = df_all[df_all['player'] == player_name][pca_columns]
    if player_row.empty:
        print(f"No PCA data available for player '{player_name}'.")
        return None

    # Calculate Euclidean distances to all other players
    distances = cdist(player_row, df_all[pca_columns], metric='euclidean').flatten()

    # Add distances to the DataFrame
    df_all['distance'] = distances

    # Sort by distance and exclude the selected player
    closest_players = df_all[df_all['player'] != player_name].sort_values(by='distance').head(10)

    # Return the top 10 closest players
    return closest_players[['player', 'distance'] + pca_columns]

# Main process
if __name__ == "__main__":
    dfs = {}
    for key in stat_keys:
        print(f"--- Processing {key} ---")
        df = load_and_process(key)
        df = apply_pca_kmeans(df, key)
        dfs[key] = df

    for name in dfs:
        if '90s' in dfs[name].columns:
            print(f"Filtering by 90s for: {name}")
            min_90s = dfs[name]['90s'].mean() * 0.5
            original_rows = dfs[name].shape[0]
            dfs[name] = dfs[name][dfs[name]['90s'] >= min_90s]
            print(f"Filtered from {original_rows} to {dfs[name].shape[0]} rows")

    merged_parts = []
    for name, df in dfs.items():
        required_cols = ['player', f'pca1_{name}', f'pca2_{name}', f'cluster_{name}']
        if all(col in df.columns for col in required_cols):
            print(f"Adding {name} to merge list")
            merged_parts.append(df[required_cols].copy())

    print("Merging all DataFrames...")
    df_all = reduce(lambda left, right: pd.merge(left, right, on='player', how='outer'), merged_parts)
    print("Merge complete. Final shape:", df_all.shape)

    # Select a random player
    random_player = random.choice(df_all['player'].dropna().unique())
    print(f"Randomly selected player: {random_player}")

    # Find the top 10 closest players to the random player
    closest_players = find_closest_players(df_all, random_player)
    if closest_players is not None:
        print(f"Top 10 closest players to '{random_player}':")
        print(closest_players)
