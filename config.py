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
"""