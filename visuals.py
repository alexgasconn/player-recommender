# visuals.py
import plotly.graph_objects as go


def create_radar_chart(stat_group_means, player_name, comparison_stats=None):
    if not stat_group_means:
        fig = go.Figure()
        fig.update_layout(title="No radar data available")
        return fig

    categories = list(stat_group_means.keys())
    values = [float(stat_group_means[c]) for c in categories]

    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            name=player_name,
            line=dict(width=2),
            opacity=0.65,
        )
    )

    if comparison_stats:
        for comp_player, comp_stats in comparison_stats.items():
            comp_values = [float(comp_stats.get(cat, 50.0))
                           for cat in categories]
            comp_values_closed = comp_values + [comp_values[0]]
            fig.add_trace(
                go.Scatterpolar(
                    r=comp_values_closed,
                    theta=categories_closed,
                    fill="none",
                    name=comp_player,
                    line=dict(width=2),
                )
            )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%"),
        ),
        margin=dict(l=30, r=30, t=60, b=20),
        showlegend=True,
        title=f"Percentile Radar: {player_name}",
    )

    return fig
