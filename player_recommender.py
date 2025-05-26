import streamlit as st
from config import urls, stat_keys, custom_styles
from data_utils import load_all_stat_dfs
from analysis import (
    find_closest_players_per_stat,
    compute_stat_group_means,
    get_position_weights
)
from visuals import create_radar_chart
import types

st.set_page_config(
    page_title="FBRef Player Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(custom_styles, unsafe_allow_html=True)


def main():
    st.title("FBRef Player Recommender System")
    st.markdown(
        "Find players who perform similarly across different stat categories based on PCA proximity.")
    st.markdown("---")

    with st.spinner("Loading player data and computing PCA + Clustering..."):
        dfs = load_all_stat_dfs()
        # Filtrar jugadores con al menos el 25% del promedio general de '90s'
        general_90s = next(iter(dfs.values()))['90s'].mean()
        threshold_90s = general_90s * 0.25
        dfs = {
            key: df[df['90s'] >= threshold_90s] if '90s' in df.columns else df
            for key, df in dfs.items()
        }

    first_df = next(iter(dfs.values()))

    st.sidebar.markdown("### Filters")
    positions = sorted(set(first_df['pos'].dropna()))
    competitions = sorted(set(first_df['comp'].dropna()))
    nations = sorted(set(first_df['nation'].dropna()))

    selected_positions = st.sidebar.multiselect(
        "Filter by Position", positions, default=positions, key="filter_pos")
    selected_competitions = st.sidebar.multiselect(
        "Filter by Competition", competitions, default=competitions, key="filter_comp")
    selected_nations = st.sidebar.multiselect(
        "Filter by Nation", nations, default=nations, key="filter_nat")

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
        closest_players_per_stat, combined_df = find_closest_players_per_stat(
            dfs, selected_player, position_weights)

        col1, col2 = st.columns(2)
        with col1:
            show_columns = ['player', 'pos', 'age', 'squad', 'similarity']
            st.markdown(
                '<p class="stat-title">Most Similar Players Overall</p>', unsafe_allow_html=True)
            st.dataframe(combined_df[show_columns], use_container_width=True)

        st.markdown(
            '<p class="stat-title">Compare with Other Players</p>', unsafe_allow_html=True)
        top_10_players = combined_df['player'].tolist()
        selected_comparison_players = st.multiselect(
            "Select players to compare:",
            options=top_10_players,
            default=[],
            help="Select one or more players to compare their statistics on the radar chart."
        )

        with col2:
            st.markdown('<p class="stat-title">Radar Chart</p>',
                        unsafe_allow_html=True)
            stat_group_means = compute_stat_group_means(dfs, selected_player)

            comparison_stats = {}
            for player in selected_comparison_players:
                comparison_stats[player] = compute_stat_group_means(
                    dfs, player)

            radar_chart = create_radar_chart(
                stat_group_means, selected_player, comparison_stats)
            st.plotly_chart(radar_chart, use_container_width=True)

        # Use tabs for closest players per stat group
        stat_names = list(closest_players_per_stat.keys())
        if stat_names:
            tab_objs = st.tabs(
                [f"{stat_name.title()}" for stat_name in stat_names])
            for tab, stat_name in zip(tab_objs, stat_names):
                with tab:
                    st.markdown(
                        f'<p class="stat-title">Closest Players in {stat_name.title()}</p>', unsafe_allow_html=True)
                    st.dataframe(
                        closest_players_per_stat[stat_name], use_container_width=True)

    # Helper to get stat group means for a list of players


def get_stat_group_means_for_players(dfs, players, stat_group):
    return {
        player: compute_stat_group_means(dfs, player, stat_group=stat_group)
        for player in players
    }

# Patch main to add radar chart for each stat group


def main_with_stat_group_radar():
    main()
    # Access Streamlit session state to get selected player
    if "Choose a player to analyze:" in st.session_state:
        selected_player = st.session_state["Choose a player to analyze:"]
        if selected_player:
            dfs = load_all_stat_dfs()
            position_weights = get_position_weights(selected_player, dfs)
            closest_players_per_stat, _ = find_closest_players_per_stat(
                dfs, selected_player, position_weights)
            stat_names = list(closest_players_per_stat.keys())
            if stat_names:
                tab_objs = st.tabs(
                    [f"Radar: {stat_name.title()}" for stat_name in stat_names])
                for tab, stat_name in zip(tab_objs, stat_names):
                    with tab:
                        # Get top 3 closest players for radar chart
                        top_players = closest_players_per_stat[stat_name]['player'].head(
                            3).tolist()
                        stat_group_means = compute_stat_group_means(
                            dfs, selected_player, stat_group=stat_name)
                        comparison_stats = get_stat_group_means_for_players(
                            dfs, top_players, stat_group=stat_name)
                        radar_chart = create_radar_chart(
                            stat_group_means, selected_player, comparison_stats, stat_group=stat_name)
                        st.plotly_chart(radar_chart, use_container_width=True)


main_with_stat_group_radar = types.FunctionType(
    main_with_stat_group_radar.__code__, globals())

if __name__ == "__main__":
    main_with_stat_group_radar()
