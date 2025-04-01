# fbref_analysis.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from functools import reduce
from scipy.spatial.distance import cdist
import streamlit as st
import plotly.graph_objects as go

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="FBRef Player Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------ STYLES ------------------
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .sub-font { font-size:18px !important; }
    .stat-title { font-size:20px !important; font-weight:bold; color: #4CAF50; margin-top: 2em; }
    .highlight-box { 
        background-color: #f0f9ff;
        padding: 1em;
        border-radius: 10px;
        border-left: 4px solid #2e86de;
        margin-bottom: 1em;
    }
    .centered {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ URLS ------------------
urls = {
    "standard_stats": "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats",
    "shooting": "https://fbref.com/en/comps/Big5/shooting/players/Big-5-European-Leagues-Stats",
    "passing": "https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats",
    "pass_types": "https://fbref.com/en/comps/Big5/passing_types/players/Big-5-European-Leagues-Stats",
    "goal_and_shot_creation": "https://fbref.com/en/comps/Big5/gca/players/Big-5-European-Leagues-Stats",
    "defensive_actions": "https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Stats",
    "possession": "https://fbref.com/en/comps/Big5/possession/players/Big-5-European-Leagues-Stats",
    "playing_time": "https://fbref.com/en/comps/Big5/playingtime/players/Big-5-European-Leagues-Stats",
    "miscellaneous_stats": "https://fbref.com/en/comps/Big5/misc/players/Big-5-European-Leagues-Stats"
}

stat_keys = [
    "standard_stats",
    "shooting",
    "passing",
    "pass_types",
    "goal_and_shot_creation",
    "defensive_actions",
    "possession",
    "playing_time",
    "miscellaneous_stats"
]

# ------------------ DATA PROCESSING ------------------

@st.cache_data
def load_and_process(url_key):
    """
    Load and process data for a given stats group.

    Args:
        url_key (str): The key for the stats group URL.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    df = pd.read_html(urls[url_key])[0]
    df.columns = df.columns.droplevel()
    df.columns = df.columns.str.strip().str.lower()

    categorical_cols = ['player', 'nation', 'pos', 'squad', 'comp', 'age', 'born', 'matches']
    for col in df.columns:
        if col not in categorical_cols:
            # Ensure the column is a valid Series before converting
            if isinstance(df[col], pd.Series):
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing numeric values with the median
    df.fillna(df.median(numeric_only=True), inplace=True)
    # Drop columns that are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def apply_pca_kmeans(df, name):
    df_numerical = df.select_dtypes(include=np.number)
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_numerical)

    df[f'pca1_{name}'] = df_pca[:, 0]
    df[f'pca2_{name}'] = df_pca[:, 1]

    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(df_pca)
    df[f'cluster_{name}'] = kmeans.labels_

    return df

def find_closest_players_per_stat(dfs, player_name, position_weights):
    """
    Find the top 10 closest players for each stat category separately and apply position-based weights.

    Args:
        dfs (dict): Dictionary of DataFrames for each stat category.
        player_name (str): The name of the player to compare against.
        position_weights (dict): Weights for each stat group based on the player's position.

    Returns:
        dict: A dictionary where keys are stat categories and values are DataFrames of the top 10 closest players.
        pd.DataFrame: A DataFrame with the combined weighted distances for all players.
    """
    closest_players_per_stat = {}
    combined_distances = {}

    # Get the selected player's position
    first_df = next(iter(dfs.values()))
    player_position = first_df[first_df['player'] == player_name]['pos'].values[0]

    for stat_name, df in dfs.items():
        if player_name not in df['player'].values:
            st.write(f"Player '{player_name}' not found in {stat_name}. Skipping...")
            continue

        # Get PCA columns for this stat
        pca_columns = [col for col in df.columns if col.startswith('pca')]
        if not pca_columns:
            st.write(f"No PCA data available for {stat_name}. Skipping...")
            continue

        # Get the player's PCA row
        player_row = df[df['player'] == player_name][pca_columns]
        if player_row.empty:
            st.write(f"No PCA data available for player '{player_name}' in {stat_name}. Skipping...")
            continue

        # Compute distances
        distances = cdist(player_row, df[pca_columns], metric='euclidean').flatten()
        df['distance'] = distances

        # Apply position-based weight to the distances
        weight = position_weights.get(stat_name, 1.0)
        df['weighted_distance'] = df['distance'] * weight

        # Apply position-based bonus/penalty
        def position_bonus(other_position):
            if player_position == other_position:  # Same position
                return -0.1  # Strong bonus
            elif (player_position == "D" and other_position == "M") or \
                 (player_position == "M" and other_position in ["D", "F"]) or \
                 (player_position == "F" and other_position == "M"):  # Adjacent positions
                return -0.05  # Moderate bonus
            else:  # Far positions
                return 0.1  # Penalty

        df['position_bonus'] = df['pos'].apply(position_bonus)
        df['final_distance'] = df['weighted_distance'] + df['position_bonus']

        # Get the top 10 closest players
        closest_players = df[df['player'] != player_name].sort_values(by='final_distance').head(10)
        closest_players_per_stat[stat_name] = closest_players[['player', 'pos', 'age', 'squad', 'final_distance']]

        # Add weighted distances to the combined distances
        for _, row in closest_players.iterrows():
            player = row['player']
            combined_distances[player] = combined_distances.get(player, 0) + row['final_distance']

    # Create a final DataFrame for combined distances
    combined_df = pd.DataFrame(list(combined_distances.items()), columns=['player', 'total_weighted_distance'])

    # Merge with one of the original DataFrames to get additional details
    combined_df = combined_df.merge(first_df[['player', 'pos', 'age', 'squad']], on='player', how='left')

    # Sort by total weighted distance and return the top 10
    combined_df = combined_df.sort_values(by='total_weighted_distance').head(10)

    return closest_players_per_stat, combined_df

def compute_stat_group_means(dfs, player_name):
    """
    Compute the percentile rank of the selected player's mean for each stats group.

    Args:
        dfs (dict): Dictionary of DataFrames for each stats group.
        player_name (str): The name of the player to analyze.

    Returns:
        dict: A dictionary where keys are stats group names and values are percentile ranks (0 to 100).
    """
    stat_group_means = {}

    for stat_name, df in dfs.items():
        if player_name not in df['player'].values:
            continue  # Skip if the player is not in this stat group

        # Get numerical columns
        numerical_cols = df.select_dtypes(include=np.number).columns

        if numerical_cols.empty:
            continue  # Skip if there are no numerical columns

        # Get the player's row
        player_row = df[df['player'] == player_name][numerical_cols]

        if player_row.empty:
            continue  # Skip if the player's data is missing

        # Compute the mean of all numerical features for all players
        group_means = df[numerical_cols].mean(axis=1)

        # Compute the mean for the selected player
        player_mean = player_row.mean(axis=1).values[0]

        # Compute the percentile rank of the player's mean
        percentile_rank = (group_means < player_mean).mean() * 100

        stat_group_means[stat_name] = percentile_rank

    return stat_group_means


def create_radar_chart(stat_group_means, player_name):
    """
    Create a radar chart for the selected player's stats group means.

    Args:
        stat_group_means (dict): A dictionary of scaled means for each stats group.
        player_name (str): The name of the player to analyze.

    Returns:
        plotly.graph_objects.Figure: A radar chart figure.
    """
    categories = list(stat_group_means.keys())
    values = list(stat_group_means.values())

    # Close the radar chart by repeating the first value
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=player_name
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[25, 100]
            )
        ),
        showlegend=True,
        title=f"Radar Chart for {player_name}"
    )

    return fig

# Define position-based weights
def get_position_weights(selected_player, dfs):
    """
    Define position-based weights for each stat group based on the player's position.

    Args:
        selected_player (str): The name of the selected player.
        dfs (dict): Dictionary of DataFrames for each stats group.

    Returns:
        dict: A dictionary of weights for each stat group.
    """
    # Get the player's position from the first DataFrame
    first_df = next(iter(dfs.values()))
    player_position = first_df[first_df['player'] == selected_player]['pos'].values[0]

    # Default weights
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

    # Adjust weights based on position
    if "F" in player_position:  # Forwards
        weights.update({
            "shooting": 1.5,
            "goal_and_shot_creation": 1.5,
            "passing": 1.2,
            "defensive_actions": 0.5,
        })
    elif "M" in player_position:  # Midfielders
        weights.update({
            "passing": 1.5,
            "possession": 1.5,
            "goal_and_shot_creation": 1.3,
            "shooting": 1.0,
            "defensive_actions": 1.0,
        })
    elif "D" in player_position:  # Defenders
        weights.update({
            "defensive_actions": 1.5,
            "possession": 1.3,
            "playing_time": 1.2,
            "shooting": 0.5,
            "goal_and_shot_creation": 0.5,
        })

    return weights

# ------------------ MAIN APP ------------------

def main():
    st.title("⚽ FBRef Player Recommender System")
    st.markdown("Find players who perform similarly across different stat categories based on PCA proximity.")
    st.markdown("---")

    # Load and process data with a progress bar
    dfs = {}
    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i, key in enumerate(stat_keys):
        progress_text.text(f"Loading and processing: {key}")
        df = load_and_process(key)
        df = apply_pca_kmeans(df, key)
        dfs[key] = df
        progress_bar.progress((i + 1) / len(stat_keys))

    # Clear the progress bar and text after loading is complete
    progress_text.empty()
    progress_bar.empty()

    # Define `first_df` as the first DataFrame in `dfs`
    first_df = next(iter(dfs.values()))

    # Sidebar Filters
    st.sidebar.markdown("### Filters")
    positions = sorted(set(first_df['pos'].dropna()))
    competitions = sorted(set(first_df['comp'].dropna()))
    nations = sorted(set(first_df['nation'].dropna()))

    selected_positions = st.sidebar.multiselect("Filter by Position", positions, default=positions)
    selected_competitions = st.sidebar.multiselect("Filter by Competition", competitions, default=competitions)
    selected_nations = st.sidebar.multiselect("Filter by Nation", nations, default=nations)

    # Filter the player list based on the selected filters
    filtered_df = first_df[
        (first_df['pos'].isin(selected_positions)) &
        (first_df['comp'].isin(selected_competitions)) &
        (first_df['nation'].isin(selected_nations))
    ]
    all_players = set(filtered_df['player'].dropna())

    # Player Selection
    player_list = sorted(all_players)
    selected_player = st.selectbox("Choose a player to analyze:", player_list)

    # Define position-based weights
    position_weights = get_position_weights(selected_player, dfs)

    # Show closest players
    if selected_player:
        closest_players_per_stat, combined_df = find_closest_players_per_stat(dfs, selected_player, position_weights)

        # Display combined distances and radar chart side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="stat-title">🏆 Most Similar Players Overall</p>', unsafe_allow_html=True)
            st.dataframe(combined_df, use_container_width=True)
        with col2:
            st.markdown('<p class="stat-title">📊 Radar Chart</p>', unsafe_allow_html=True)
            stat_group_means = compute_stat_group_means(dfs, selected_player)
            radar_chart = create_radar_chart(stat_group_means, selected_player)
            st.plotly_chart(radar_chart, use_container_width=True)

        # Toggle visibility of distance tables for each stat group
        for stat_name, closest_players in closest_players_per_stat.items():
            if st.checkbox(f"Show closest players in {stat_name.title()}", key=stat_name):
                st.markdown(f'<p class="stat-title">🔎 Closest Players in {stat_name.title()}</p>', unsafe_allow_html=True)
                st.dataframe(closest_players, use_container_width=True)


if __name__ == "__main__":
    main()
