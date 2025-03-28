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

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="FBRef Player Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
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

# ------------------ DATA PROCESSING ------------------

@st.cache_data
def load_and_process(url_key):
    st.write(f"Loading data for: {url_key}")
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
                st.write(f"Skipping column '{col}' as it is not a valid Series.")

    # Fill missing numeric values with the median
    df.fillna(df.median(numeric_only=True), inplace=True)
    # Drop columns that are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    st.write(f"Finished processing: {url_key} (rows: {df.shape[0]}, cols: {df.shape[1]})")
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

def find_closest_players_per_stat(dfs, player_name):
    closest_players_per_stat = {}

    for stat_name, df in dfs.items():
        if player_name not in df['player'].values:
            continue

        pca_columns = [col for col in df.columns if col.startswith('pca')]
        player_row = df[df['player'] == player_name][pca_columns]
        if player_row.empty:
            continue

        distances = cdist(player_row, df[pca_columns], metric='euclidean').flatten()
        df['distance'] = distances
        closest_players = df[df['player'] != player_name].sort_values(by='distance').head(10)
        closest_players_per_stat[stat_name] = closest_players[['player', 'pos', 'age', 'squad', 'distance'] + pca_columns]

    return closest_players_per_stat

# ------------------ MAIN APP ------------------

def main():
    st.title("⚽ FBRef Player Recommender System")
    st.markdown("Find players who perform similarly across different stat categories based on PCA proximity.")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.markdown("""
    - Data Source: [FBRef](https://fbref.com/)
    - Clustering: PCA + KMeans
    - Stat Categories: Shooting, Passing, GCA, Defense, etc.
    """)
    st.sidebar.markdown("Built with ❤️ using Streamlit")

    # Load and process
    st.markdown('<p class="big-font">1. Loading Player Data</p>', unsafe_allow_html=True)
    dfs = {}
    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i, key in enumerate(stat_keys):
        progress_text.text(f"Loading and processing: {key}")
        df = load_and_process(key)
        df = apply_pca_kmeans(df, key)
        dfs[key] = df
        progress_bar.progress((i + 1) / len(stat_keys))

    progress_text.text("✅ All data loaded and processed.")
    progress_bar.empty()

    # Select player
    st.markdown('<p class="big-font">2. Select a Player</p>', unsafe_allow_html=True)
    all_players = set()
    for df in dfs.values():
        all_players.update(df['player'].dropna().unique())

    player_list = sorted(all_players)
    selected_player = st.selectbox("Choose a player to analyze:", player_list)

    # Show closest players
    if selected_player:
        st.markdown('<div class="highlight-box">Analyzing player: <b>{}</b></div>'.format(selected_player), unsafe_allow_html=True)
        closest_players_per_stat = find_closest_players_per_stat(dfs, selected_player)

        for stat_name, closest_players in closest_players_per_stat.items():
            st.markdown(f'<p class="stat-title">🔎 Closest Players in {stat_name.title()}</p>', unsafe_allow_html=True)
            st.dataframe(closest_players, use_container_width=True)

if __name__ == "__main__":
    main()
