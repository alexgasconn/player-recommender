# app.py
import streamlit as st
from config import urls, stat_keys, custom_styles
from data_utils import load_all_stat_dfs
from analysis import (
    find_closest_players_per_stat,
    compute_stat_group_means,
    get_position_weights
)
from visuals import create_radar_chart

st.set_page_config(
    page_title="FBRef Player Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(custom_styles, unsafe_allow_html=True)

def main():
    st.title("FBRef Player Recommender System")
    st.markdown("Find players who perform similarly across different stat categories based on PCA proximity.")
    st.markdown("---")

    with st.spinner("Loading player data and computing PCA + Clustering..."):
        dfs = load_all_stat_dfs()

    first_df = next(iter(dfs.values()))

    st.sidebar.markdown("### Filters")
    positions = sorted(set(first_df['pos'].dropna()))
    competitions = sorted(set(first_df['comp'].dropna()))
    nations = sorted(set(first_df['nation'].dropna()))

    selected_positions = st.sidebar.multiselect("Filter by Position", positions, default=positions)
    selected_competitions = st.sidebar.multiselect("Filter by Competition", competitions, default=competitions)
    selected_nations = st.sidebar.multiselect("Filter by Nation", nations, default=nations)

    filtered_df = first_df[
        (first_df['pos'].isin(selected_positions)) &
        (first_df['comp'].isin(selected_competitions)) &
        (first_df['nation'].isin(selected_nations))
    ]
    all_players = set(filtered_df['player'].dropna())

    player_list = sorted(all_players)
    selected_player = st.selectbox("Choose a player to analyze:", player_list)

    if selected_player:
        position_weights = get_position_weights(selected_player, dfs)
        closest_players_per_stat, combined_df = find_closest_players_per_stat(dfs, selected_player, position_weights)

        col1, col2 = st.columns(2)
        with col1:
            show_columns = ['player', 'pos', 'age', 'squad', 'similarity']
            st.markdown('<p class="stat-title">Most Similar Players Overall</p>', unsafe_allow_html=True)
            st.dataframe(combined_df[show_columns], use_container_width=True)

        st.markdown('<p class="stat-title">Compare with Other Players</p>', unsafe_allow_html=True)
        top_10_players = combined_df['player'].tolist()
        selected_comparison_players = st.multiselect(
            "Select players to compare:",
            options=top_10_players,
            default=[],
            help="Select one or more players to compare their statistics on the radar chart."
        )

        with col2:
            st.markdown('<p class="stat-title">Radar Chart</p>', unsafe_allow_html=True)
            stat_group_means = compute_stat_group_means(dfs, selected_player)

            comparison_stats = {}
            for player in selected_comparison_players:
                comparison_stats[player] = compute_stat_group_means(dfs, player)

            radar_chart = create_radar_chart(stat_group_means, selected_player, comparison_stats)
            st.plotly_chart(radar_chart, use_container_width=True)

        for stat_name, closest_players in closest_players_per_stat.items():
            if st.checkbox(f"Show closest players in {stat_name.title()}", key=stat_name):
                st.markdown(f'<p class="stat-title">Closest Players in {stat_name.title()}</p>', unsafe_allow_html=True)
                st.dataframe(closest_players, use_container_width=True)

if __name__ == "__main__":
    main()
