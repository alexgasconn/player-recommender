import streamlit as st
from config import custom_styles
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
    initial_sidebar_state="expanded"
)

st.markdown(custom_styles, unsafe_allow_html=True)


def main():
    st.title("FBRef Player Recommender")
    st.markdown(
        "Player discovery with multi-stat similarity, position-aware weighting, and interactive scouting filters."
    )
    st.caption(
        "Data source: FBRef Big-5 public tables. Similarity uses PCA space per stat group and weighted distance aggregation."
    )
    st.markdown("---")

    try:
        with st.spinner("Loading player data and computing PCA + Clustering..."):
            dfs = load_all_stat_dfs()
    except Exception as exc:
        st.error(
            "Data loading failed (FBRef request error). Please retry in a moment; "
            "the source may be temporarily rate-limited."
        )
        st.caption(str(exc))
        return

    first_df = next(iter(dfs.values()))

    st.sidebar.markdown("### Filters")
    positions = sorted(set(first_df['pos'].dropna()))
    competitions = sorted(set(first_df['comp'].dropna()))
    nations = sorted(set(first_df['nation'].dropna()))

    age_min = int(first_df['age_year'].min()
                  ) if 'age_year' in first_df.columns else 16
    age_max = int(first_df['age_year'].max()
                  ) if 'age_year' in first_df.columns else 40
    raw_90s_max = float(first_df['raw_90s'].max()
                        ) if 'raw_90s' in first_df.columns else 50.0

    selected_positions = st.sidebar.multiselect(
        "Filter by Position", positions, default=positions, key="filter_pos")
    selected_competitions = st.sidebar.multiselect(
        "Filter by Competition", competitions, default=competitions, key="filter_comp")
    selected_nations = st.sidebar.multiselect(
        "Filter by Nation", nations, default=nations, key="filter_nat")
    selected_age = st.sidebar.slider(
        "Age Range",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
    )
    min_90s = st.sidebar.slider(
        "Minimum 90s Played",
        min_value=0.0,
        max_value=raw_90s_max,
        value=min(5.0, raw_90s_max),
        step=0.5,
    )
    same_position_only = st.sidebar.checkbox(
        "Only same main position",
        value=False,
    )
    quality_mode = st.sidebar.selectbox(
        "Recommendation quality profile",
        options=["balanced", "strict", "exploratory"],
        index=0,
        help="Strict penalizes age/minutes mismatch more. Exploratory allows broader profile discovery.",
    )
    prefer_same_comp = st.sidebar.checkbox(
        "Prioritize same competition",
        value=True,
        help="Adds a small bonus to players from the same competition as the selected player.",
    )
    top_k = st.sidebar.slider(
        "Number of recommendations", min_value=5, max_value=20, value=10, step=1)

    filtered_df = first_df[
        (first_df['pos'].isin(selected_positions)) &
        (first_df['comp'].isin(selected_competitions)) &
        (first_df['nation'].isin(selected_nations)) &
        (first_df['age_year'].between(selected_age[0], selected_age[1]) if 'age_year' in first_df.columns else True) &
        ((first_df['raw_90s'] >= min_90s)
         if 'raw_90s' in first_df.columns else True)
    ]

    if filtered_df.empty:
        st.warning(
            "No players match the current filters. Relax at least one filter in the sidebar.")
        return

    all_players = set(filtered_df['player'].dropna())

    player_list = sorted(all_players)
    col_player, col_compare = st.columns([2, 3])
    with col_player:
        selected_player = st.selectbox(
            "Choose a player to analyze:", player_list)
    with col_compare:
        selected_comparison_players = st.multiselect(
            "Select players to compare:",
            options=[p for p in player_list if p != selected_player],
            default=[],
            help="Select one or more players to compare their statistics on the radar chart."
        )

    if selected_player:
        position_weights = get_position_weights(selected_player, dfs)
        closest_players_per_stat, combined_df = find_closest_players_per_stat(
            dfs,
            selected_player,
            position_weights,
            candidate_players=all_players,
            top_k=top_k,
            same_position_only=same_position_only,
            quality_mode=quality_mode,
            prefer_same_comp=prefer_same_comp,
        )

        if combined_df.empty:
            st.error(
                "No recommendations available for this player with the current filters.")
            return

        top_recommended = combined_df['player'].tolist()

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Candidate Pool", len(player_list))
        kpi2.metric("Recommended Players", len(combined_df))
        kpi3.metric("Stat Groups Used", len(closest_players_per_stat))

        tab_overview, tab_by_group, tab_radar = st.tabs(
            ["Overview", "By Stat Group", "Radar Compare"])

        with tab_overview:
            st.markdown(
                '<p class="stat-title">Most Similar Players (Overall)</p>', unsafe_allow_html=True)
            display_df = combined_df.copy()
            display_df['similarity'] = display_df['similarity'].map(
                lambda x: f"{x:.2f}%")
            st.dataframe(
                display_df[[
                    'player',
                    'pos',
                    'age',
                    'squad',
                    'comp',
                    'similarity',
                    'confidence',
                    'groups_covered',
                    'coverage_pct',
                ]],
                use_container_width=True,
                hide_index=True,
            )

        with tab_by_group:
            stat_names = list(closest_players_per_stat.keys())
            if not stat_names:
                st.info(
                    "No stat-group recommendations available for the current selection.")
            else:
                selected_stat = st.selectbox(
                    "Stat group",
                    stat_names,
                    format_func=lambda x: x.replace("_", " ").title(),
                )
                st.dataframe(
                    closest_players_per_stat[selected_stat],
                    use_container_width=True,
                    hide_index=True,
                )

        with tab_radar:
            base_stats = compute_stat_group_means(
                dfs, selected_player, reference_players=all_players)
            comparison_pool = selected_comparison_players or top_recommended[:3]
            comparison_stats = get_stat_group_means_for_players(
                dfs, comparison_pool, reference_players=all_players)
            radar_chart = create_radar_chart(
                base_stats, selected_player, comparison_stats)
            st.plotly_chart(radar_chart, use_container_width=True)

            st.caption(
                "Radar values represent average percentile per stat group (0-100). "
                "If no comparison players are selected, the top-3 recommendations are used by default."
            )

        with st.expander("How similarity works"):
            st.markdown(
                """
                - Each stat group is projected to PCA space.
                - Distances are computed in that space and weighted by role-specific importance.
                - A position compatibility factor penalizes distant roles.
                - Context factors (minutes, age proximity and optional competition match) improve ranking quality.
                - Overall ranking uses average weighted distance across available stat groups.
                """
            )


def get_stat_group_means_for_players(dfs, players, stat_group=None, reference_players=None):
    return {
        player: compute_stat_group_means(
            dfs, player, stat_group=stat_group, reference_players=reference_players)
        for player in players
    }


if __name__ == "__main__":
    main()
