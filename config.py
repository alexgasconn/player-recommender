# config.py

# URLs de FBRef para scraping
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

# Lista de claves de estadísticas
stat_keys = list(urls.keys())

# Estilos CSS para Streamlit
custom_styles = """
<style>
:root {
    --bg-soft: #f6f8fa;
    --panel-border: #d0d7de;
    --accent: #0f766e;
    --accent-soft: #e6fffa;
}

.main .block-container {
    padding-top: 1.5rem;
}

.stApp {
    background: radial-gradient(circle at top right, #ffffff 0%, var(--bg-soft) 75%);
}

.stat-title {
    font-size: 1.05rem !important;
    font-weight: 700;
    color: var(--accent);
    margin-top: 0.75rem;
    margin-bottom: 0.35rem;
}

div[data-testid="stMetric"] {
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    background: #ffffff;
    padding: 0.5rem 0.75rem;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    overflow: hidden;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    border: 1px solid var(--panel-border);
    padding: 0.35rem 0.8rem;
    background: #ffffff;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: #0b4f49;
}
</style>
"""
