# ⚽ FBRef Player Recommender

A Streamlit web app that recommends players with similar profiles using performance statistics from [FBRef](https://fbref.com/). The tool performs PCA + KMeans clustering on various stat groups and visualizes the most similar players to a given player, along with percentile-based radar charts.

## 🔍 Features

- Scrapes and processes player statistics from FBRef across six categories:
  - Standard Stats
  - Shooting
  - Passing
  - Defense
  - Goal-Creating Actions (GCA)
  - Miscellaneous
- Standardizes and applies PCA for dimensionality reduction
- Clusters players using KMeans (n=10)
- Finds the 10 most similar players per stat group and overall
- Computes percentile-based radar chart of player profile
- Clean, interactive UI built with Streamlit and Plotly

## 📊 Example

<div align="center">
  <img src="https://raw.githubusercontent.com/alexgasconn/player-recommender/main/assets/example_dashboard.png" width="80%" alt="App screenshot">
</div>

## 🛠️ How It Works

1. Loads and processes FBRef data using `pandas` and `sklearn`
2. Applies PCA and clusters players into groups using KMeans
3. Computes weighted Euclidean distances between players in each stat group
4. Combines distances using position-based weights
5. Outputs most similar players and radar chart of performance profile

## 🧰 Tech Stack

- Python
- Streamlit
- Pandas / NumPy / Scikit-learn
- Plotly
- BeautifulSoup (FBRef scraping)

## 🚀 Run the App

```bash
# Clone the repository
git clone https://github.com/alexgasconn/player-recommender.git
cd player-recommender

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run fbref_analysis.py
