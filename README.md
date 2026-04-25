# FBRef Player Recommender

Production-oriented Streamlit application for football player scouting and profile matching using FBRef Big-5 player tables. The app builds role-aware similarity rankings from multi-domain performance data and exposes interactive filtering and comparison workflows for analysts, coaches, and recruitment teams.

## 1) Project Overview

### What this app does

This app helps users find players with a similar statistical profile to a selected target player. It combines multiple data domains (shooting, passing, defensive actions, possession, etc.) into a unified recommendation pipeline and produces:

- Overall similarity ranking (cross-domain)
- Per-stat-group nearest players
- Radar-based profile comparisons in percentile space

### Who it is for

- Football recruitment analysts
- Coaching staff and tactical analysts
- Data practitioners building shortlist workflows
- Fans who want evidence-based player comparisons

### Problem it solves

Raw football stats are high-dimensional, noisy, and role-dependent. Naive comparisons can be misleading. The app addresses this by:

- Normalizing multi-source feature spaces
- Comparing players in reduced-dimensional latent spaces (PCA)
- Applying position-aware weighting to reduce role mismatch
- Providing transparent, interactive filtering controls

### Data sources

Current implementation uses FBRef public Big-5 league player tables, fetched via HTML table ingestion (`pandas.read_html`) from multiple endpoints:

- Standard stats
- Shooting
- Passing
- Pass types
- Goal and shot creation
- Defensive actions
- Possession
- Playing time
- Miscellaneous stats

Data format is tabular HTML and transformed into in-memory Pandas DataFrames.

### Key features (UI sections, tabs, filters)

- Sidebar filters:
  - Position
  - Competition
  - Nation
  - Age range
  - Minimum 90s played
  - Same-main-position toggle
  - Top-K recommendation size
- Main sections:
  - Overview tab: overall recommendation table
  - By Stat Group tab: nearest players inside one selected domain
  - Radar Compare tab: percentile profile comparison between selected players
- Explanatory panel documenting the recommendation logic

---

## 2) Architecture and System Design

### Module layout

- `player_recommender.py`: Streamlit app entrypoint and UI orchestration
- `config.py`: endpoint registry, stat-group keys, and style definitions
- `data_utils.py`: data ingestion, cleaning, feature preparation, PCA, clustering
- `analysis.py`: similarity logic, aggregation, role-based weighting, percentile scoring
- `visuals.py`: Plotly chart builders (radar)

### End-to-end data flow

1. Ingestion
   - Fetch FBRef tables with `pandas.read_html`
2. Cleaning and harmonization
   - Flatten multi-index headers
   - Remove repeated header rows
   - Coerce numeric fields
   - Parse position and age
   - Impute numeric missing values (median)
3. Feature engineering
   - Preserve `raw_90s` for realistic filtering
   - Standardize numeric features using `StandardScaler`
4. Representation learning
   - Apply PCA per stat group with adaptive component count
5. Group structure
   - Run KMeans per stat group (adaptive cluster count)
6. Similarity scoring
   - Compute Euclidean distances in PCA space
   - Apply stat-group role weights
   - Apply position compatibility factor
   - Aggregate into cross-group final ranking
7. Presentation
   - Tables + radar chart with interactive filtering and comparison

### ETL pipeline detail

- Ingestion:
  - Source URLs are centralized in `config.py`
  - Each stat-group table is loaded independently to isolate failures
- Cleaning:
  - Headers normalized to lowercase
  - Non-numeric noise coerced to `NaN` and imputed
  - Positions reduced to primary role token
  - `age_year` extracted from FBRef age format (for slider filtering)
- Transformation:
  - Feature scaling done after cleaning and before PCA
- Feature extraction:
  - PCA latent components per stat group (`pca_<group>_<i>`)
  - Cluster labels (`cluster_<group>`) for downstream analysis
- Serving/visualization layer:
  - Cached preprocessing functions reduce repeated expensive operations

### State management

State is managed through Streamlit widget state and deterministic recomputation:

- Sidebar widgets define a dynamic candidate pool
- Target player and optional compare players drive computation graph
- Output tables/charts update reactively on user input changes

### Chart generation

- Plotly radar chart is generated from percentile scores in `[0, 100]`
- Comparison series are aligned to identical category order
- Missing category values for comparison players are safely defaulted

### Filter mechanics

- Filters are applied to a canonical DataFrame (first stat-group frame)
- Candidate player set is then passed into recommendation engine
- Recommendation logic runs only over filtered candidate set, improving relevance

### API calls, limits, caching

- External calls: FBRef public webpages (HTTP GET via `pandas.read_html` internals)
- Authentication: none required for current source
- Rate limiting: not explicit in code; practical recommendation is to avoid forced refresh bursts
- Caching: Streamlit `@st.cache_data` caches transformed DataFrames and model outputs
- Pagination: not applicable (tables loaded whole)

---

## 3) Detailed Feature Breakdown

### Sidebar filter panel

What user sees:

- Multi-select filters and sliders

Data used:

- Base DataFrame columns: `pos`, `comp`, `nation`, `age_year`, `raw_90s`

Transformations:

- Boolean mask intersection across selected dimensions

Insight:

- Restricts recommendation universe to role/context-specific market segment

Interactive elements:

- Multi-selects, numeric range sliders, top-k slider, strict-position toggle

### Overview tab

What user sees:

- Ranked nearest players table

Data used:

- Aggregated distance across all available stat groups

Behind the scenes:

- PCA-space distance per stat group
- Role-weighted and position-adjusted distance
- Average distance aggregation across groups
- Distance-to-similarity normalization

Insight:

- Fast shortlist of globally similar profiles

Interactive elements:

- Rankings update based on all active filters and selected target player

### By Stat Group tab

What user sees:

- Nearest players list within a selected stat domain

Data used:

- Stat-group-specific PCA component columns

Behind the scenes:

- Euclidean distance in selected group latent space
- Position compatibility adjustment

Insight:

- Explains *why* a player appears similar (e.g., passing-only similarity)

Interactive elements:

- Stat-group selector dropdown

### Radar Compare tab

What user sees:

- Radar chart with target player and comparison players

Data used:

- Percentile mean per stat group

Behind the scenes:

- Rank percentile for each numeric feature inside each group
- Aggregate to group-level percentile score

Insight:

- Visual profile overlap and complementary strengths

Interactive elements:

- Optional custom comparison selection; fallback to top recommendations

---

## 4) Data Engineering Details

### Cleaning steps

- Flatten FBRef multi-row headers
- Remove duplicate embedded header rows
- Standardize lowercase column names
- Numeric coercion with robust error handling
- Median imputation for numeric missing values
- Unknown fallback for critical categorical columns

### Missing values, outliers, noise

- Missing values:
  - Numeric: median imputation
  - Categorical: `Unknown`
- Outliers:
  - No hard clipping in baseline; PCA/scaling reduces extreme dominance
  - Optional future extension: robust scaling or winsorization
- Noise handling:
  - Data source is tabular event aggregates (not GPS), so no GPS jitter pipeline is needed

### Aggregations and normalization

- Standardization (`StandardScaler`) for numerical comparability
- PCA for compression and de-correlation
- Role-aware weighted distance aggregation
- Similarity score normalization to percentage scale

### Domain-specific metrics currently represented

- Shooting volume/quality proxies
- Passing volume/quality proxies
- Possession and progression proxies
- Defensive activity proxies
- Creation actions (GCA/SCA related)
- Availability/load proxy via `90s`

---

## 5) Machine Learning and Analytics

### Models used

- PCA (per stat group)
  - Purpose: dimensionality reduction and denoising
  - Benefit: distance in latent space is more stable than raw high-dimensional distance
- KMeans (per stat group)
  - Purpose: latent segmentation for profile clusters
  - Current role: precomputed structural label for exploratory use

### Feature engineering

- Per-group feature spaces built from numeric columns
- Age parsing into numeric years for filtering
- Position normalization to main role category
- Raw minutes surrogate (`raw_90s`) preserved for meaningful thresholding

### Scoring and evaluation design

- Core score: weighted Euclidean distance in PCA coordinates
- Position adjustment: multiplicative compatibility factor
- Final similarity: normalized inverse-distance percentage

### Interpretation

- Higher similarity means lower adjusted latent-space distance
- Differences by group explain stylistic similarity dimensions

### Why these methods

- PCA + distance is transparent, fast, and robust for tabular scouting data
- KMeans gives profile segmentation without labeled targets
- Position-aware weighting injects domain knowledge and improves practical relevance

---

## 6) Visualizations

### Radar chart

What it shows:

- Group-level percentile profile in `[0,100]`

How computed:

- For each group, compute per-feature percentile rank then average

Why useful:

- Easy shape-based comparison between target and alternatives

Tech:

- Plotly `Scatterpolar`

Interactivity:

- Hover values, legend toggling, dynamic player selection

### Recommendation tables

What they show:

- Ranked similar players overall and per group

How computed:

- Sorted by adjusted distance with similarity percentages

Why useful:

- Practical output for shortlist generation and manual review

---

## 7) Technical Stack

- Language: Python 3
- App framework: Streamlit
- Data: Pandas, NumPy
- ML/analytics: Scikit-learn (PCA, KMeans, StandardScaler), SciPy distance
- Visualization: Plotly
- HTML parsing: lxml (through Pandas HTML readers)

Frontend and backend model:

- Single-process Streamlit app (UI and compute in one Python service)

Deployment options:

- Local execution (default)
- Streamlit Community Cloud (recommended lightweight hosting path)

---

## 8) How to Run

### Prerequisites

- Python 3.10+ recommended
- Internet access (for FBRef table retrieval)

### Installation

```bash
git clone https://github.com/alexgasconn/player-recommender.git
cd player-recommender
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run player_recommender.py
```

### Environment variables and API keys

- No API key required in current version
- No mandatory environment variables

---

## 9) Advanced Notes

### Performance considerations

- Initial load cost dominated by remote HTML parsing and feature transforms
- Cache decorators avoid recomputation across reruns
- Candidate filtering before similarity reduces runtime under broad datasets

### Limitations

- Public table schema may change upstream and require column adaptation
- No explicit stale cache invalidation policy beyond Streamlit defaults
- Similarity is statistical, not tactical/contextual video-grounded analysis
- Current weighting is handcrafted, not learned from transfer outcomes

### Known issues

- Upstream FBRef connectivity or layout changes can temporarily break ingestion
- Extremely strict filter combinations can yield empty candidate pools

### Future improvements

- Persistent local snapshot layer (CSV/Parquet) with refresh policy
- Model explainability panel per recommendation (distance decomposition)
- Robust scaler alternatives and outlier clipping strategies
- Learned metric approaches (e.g., Mahalanobis, metric learning)
- Position-subrole granularity (FB/CB/DM/AM/Winger/CF)

---

## 10) Additional Technical Guide

For a deeper engineering document (algorithm internals, module contracts, extension patterns, deployment and observability recommendations), see:

- [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)
