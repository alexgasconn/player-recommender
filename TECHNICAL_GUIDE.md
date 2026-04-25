# Technical Guide: FBRef Player Recommender

This document provides a deep technical reference for maintainers and contributors.

## A. System Architecture Deep Dive

### A.1 Runtime topology

The application is a single Streamlit process that performs:

1. Remote data ingestion from FBRef
2. In-memory ETL and feature generation
3. Similarity analytics
4. Interactive rendering in the browser

There is no separate API server or dedicated persistence layer in the current baseline.

### A.2 Module responsibilities

- `config.py`
  - Source URL registry for all stat groups
  - UI style constants
- `data_utils.py`
  - Table ingestion and schema harmonization
  - Numeric coercion and null handling
  - Feature scaling
  - PCA and KMeans feature generation
  - Cached loading of all stat-group DataFrames
- `analysis.py`
  - Position taxonomy and compatibility functions
  - Per-group nearest-neighbor computation
  - Multi-group score aggregation
  - Group percentile profile scoring for radar
  - Role-sensitive group weighting
- `visuals.py`
  - Plotly radar generation
- `player_recommender.py`
  - UI controls, filter orchestration, and rendering logic

### A.3 Data contracts

Expected minimal columns in each stat-group DataFrame:

- Identifier/context: `player`, `pos`, `squad`, `comp`, `nation`, `age`
- Utility fields:
  - `age_year` (derived integer-like field)
  - `raw_90s` (copied from `90s` before downstream transformations)
- Generated fields:
  - `pca_<group>_<i>` for latent coordinates
  - `cluster_<group>` for unsupervised label

## B. ETL and Analytics Pipeline

### B.1 Ingestion

Source pull:

- `pd.read_html(url)` per configured stat endpoint.

Rationale:

- Fast extraction from public tabular pages.
- No custom scraper needed for baseline reliability.

### B.2 Cleaning and harmonization

Key operations:

1. Flatten multi-index headers to a single level.
2. Normalize column names (`lower`, `strip`).
3. Remove embedded duplicate header rows where `player == 'Player'`.
4. Parse age string to numeric year (`age_year`).
5. Normalize position to primary token before comma.
6. Coerce non-categorical columns to numeric.
7. Fill numeric nulls with median.
8. Fill critical categorical nulls with `Unknown`.

### B.3 Feature transformation

- Preserve `raw_90s` for downstream user filtering.
- Fit `StandardScaler` on numerical feature matrix.
- Fit PCA with adaptive component count:
  - `n_components = min(4, n_features, max(2, n_rows - 1))`

### B.4 Clustering

- Fit KMeans over PCA embedding.
- Adaptive cluster count:
  - `n_clusters = min(10, max(2, n_rows // 20))`

### B.5 Similarity computation

Per stat group:

1. Select group-specific PCA columns.
2. Compute Euclidean distance from target player to each candidate.
3. Multiply by stat-group role weight.
4. Multiply by position compatibility factor.
5. Normalize to `% similarity` by inverse min-max scaling.

Cross-group aggregation:

- For each candidate, average adjusted distance across contributing groups.
- Compute final normalized similarity.
- Return top-K nearest players.

## C. Position-Aware Recommendation Logic

### C.1 Main-position extraction

Player position strings are mapped to:

- `GK`, `D`, `M`, `F`, fallback `UNK`.

### C.2 Compatibility factors

Current heuristic factors:

- Same role: stronger similarity boost (`0.92` multiplier)
- Adjacent tactical families (e.g., M↔D/F): near-neutral (`0.98`)
- Distant roles: penalty (`1.08`)

### C.3 Stat-group weight presets

Base weight = `1.0`, then updated by target role:

- Forward-focused profiles boost shooting and creation groups.
- Midfield-focused profiles boost passing and possession.
- Defender-focused profiles boost defensive and availability-related groups.

## D. UI Design and User Workflows

### D.1 Primary workflow

1. User narrows player universe with sidebar controls.
2. User selects target player.
3. App computes weighted nearest profiles.
4. User reviews overall shortlist.
5. User inspects per-group nearest players.
6. User compares radar profiles with selected or auto-suggested peers.

### D.2 Sidebar filter semantics

- Position, competition, and nation are hard candidate constraints.
- Age and `raw_90s` prevent noisy low-sample comparisons.
- Same-position toggle enforces strict tactical comparability.
- Top-K controls output size only, not candidate universe size.

### D.3 Explainability panel

The expandable panel in the app documents the distance pipeline in plain language for non-technical users.

## E. Visualization Engineering

### E.1 Radar chart specifics

- Axis categories = stat groups.
- Radius = averaged percentile per group.
- Domain range fixed to `[0, 100]` for consistent interpretation.
- Comparison traces are aligned to identical axis order.

### E.2 Why percentiles instead of raw values

- Cross-stat comparability improves drastically.
- Units and scale mismatch are minimized.
- End users can reason in relative ranking terms.

## F. Performance and Reliability

### F.1 Caching strategy

- `@st.cache_data` used in data prep and transform functions.
- Benefits:
  - Faster reruns during widget interactions.
  - Reduced repeated network/parsing overhead.

### F.2 Failure modes

1. Upstream HTML schema drift.
2. Temporary network errors.
3. Empty output when filters are too restrictive.

### F.3 Hardening recommendations

1. Add retry with exponential backoff for remote ingestion.
2. Add schema validation checks with fallback column maps.
3. Add local data snapshot fallback (`Parquet` cache).
4. Add graceful warning banners per failed stat group.

## G. Testing and Quality Guidelines

### G.1 Unit tests to add

- Header flattening and row cleanup.
- Age parser with malformed values.
- Position mapper behavior for edge labels.
- Similarity normalization when all distances are equal.
- Candidate pool filtering correctness.

### G.2 Integration tests to add

- End-to-end load with mocked FBRef tables.
- UI-level smoke tests for non-empty recommendation results.

### G.3 Data-quality checks

- Null ratio alarms by column.
- Distribution checks for key metrics (`90s`, age).
- Duplicate player name checks by stat group.

## H. Extension Playbook

### H.1 Add a new stat group

1. Add URL in `config.py`.
2. Ensure column naming aligns with parser assumptions.
3. Include key in `stat_keys`.
4. Validate PCA columns are generated.
5. Optionally adjust role weights in `get_position_weights`.

### H.2 Introduce custom similarity metric

- Replace Euclidean with a configurable metric layer:
  - Cosine distance
  - Mahalanobis (with covariance regularization)
  - Learned metric (future supervised setup)

### H.3 Add explainability outputs

- Emit per-group contribution columns in overview table.
- Add waterfall/bar plot of weighted distance contributions.

## I. Deployment Notes

### I.1 Local

```bash
pip install -r requirements.txt
streamlit run player_recommender.py
```

### I.2 Streamlit Cloud

- Set app entrypoint: `player_recommender.py`
- Ensure `requirements.txt` is complete and pinned if reproducibility is required.

### I.3 Reproducibility

- Keep deterministic random states where possible (`random_state=42`).
- Consider pinning dependency versions in production-grade deployment.

## J. Current Limitations and Roadmap

### J.1 Limitations

- Similarity remains purely stat-profile based.
- Team style and tactical system effects are not explicitly controlled.
- No temporal trend modeling across seasons in baseline.

### J.2 Suggested roadmap

1. Multi-season longitudinal mode.
2. League-strength normalization.
3. Age-curve and contract value constraints.
4. Advanced explainability and recommendation confidence intervals.
5. Optional persistence layer for snapshots and auditability.
