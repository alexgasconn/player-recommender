# visuals.py
import plotly.graph_objects as go

def create_radar_chart(stat_group_means, player_name, comparison_stats=None):
    categories = list(stat_group_means.keys())
    values = list(stat_group_means.values())

    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=player_name
    ))

    if comparison_stats:
        for comp_player, comp_stats in comparison_stats.items():
            comp_values = list(comp_stats.values())
            comp_values.append(comp_values[0])
            fig.add_trace(go.Scatterpolar(
                r=comp_values,
                theta=categories,
                fill='none',
                name=comp_player
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title=f"Radar Chart for {player_name} and Comparisons"
    )

    return fig
