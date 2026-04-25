# data_utils.py
import pandas as pd
import numpy as np
import requests
import time
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
from config import urls, stat_keys


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _read_html_with_retry(url: str, retries: int = 3, timeout: int = 20) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url, headers=DEFAULT_HEADERS, timeout=timeout)
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            if not tables:
                raise ValueError(f"No HTML tables found in {url}")
            return tables[0]
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.2 * attempt)

    raise RuntimeError(
        f"Failed to fetch FBRef data from {url}: {last_error}") from last_error


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def _extract_age_year(age_value):
    if pd.isna(age_value):
        return np.nan
    text = str(age_value)
    return pd.to_numeric(text.split("-")[0], errors="coerce")


@st.cache_data(show_spinner=False)
def load_and_process(url_key):
    df = _read_html_with_retry(urls[url_key])
    df = _flatten_columns(df)

    if "player" in df.columns:
        df = df[df["player"].astype(str).str.lower() != "player"]

    if "age" in df.columns:
        df["age_year"] = df["age"].apply(_extract_age_year)

    if "pos" in df.columns:
        df["pos"] = df["pos"].astype(str).str.split(",").str[0].str.strip()

    categorical_cols = {"player", "nation", "pos",
                        "squad", "comp", "age", "born", "matches"}
    for col in df.columns:
        if col in categorical_cols:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(axis=1, how="all").copy()

    if "90s" in df.columns:
        df["raw_90s"] = df["90s"]

    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        medians = df[numeric_cols].median(numeric_only=True)
        df[numeric_cols] = df[numeric_cols].fillna(medians)

    for col in ["player", "nation", "pos", "squad", "comp"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def apply_pca_kmeans(df: pd.DataFrame, name: str):
    df = df.copy()
    feature_cols = [
        col
        for col in df.select_dtypes(include=np.number).columns
        if not col.startswith("pca_") and not col.startswith("cluster_")
    ]

    if len(feature_cols) < 2 or len(df) < 3:
        return df

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])

    n_components = min(4, scaled.shape[1], max(2, scaled.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=42)
    df_pca = pca.fit_transform(scaled)

    for i in range(n_components):
        df[f"pca_{name}_{i+1}"] = df_pca[:, i]

    n_clusters = min(10, max(2, len(df) // 20))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df[f"cluster_{name}"] = kmeans.fit_predict(df_pca)

    return df


@st.cache_data(show_spinner=False)
def load_all_stat_dfs():
    dfs = {}
    failed = []
    for key in stat_keys:
        try:
            df = load_and_process(key)
            dfs[key] = apply_pca_kmeans(df, key)
        except Exception:
            failed.append(key)

    if not dfs:
        failed_text = ", ".join(failed) if failed else "unknown sources"
        raise RuntimeError(
            "Unable to load player data from FBRef. "
            f"Failed stat groups: {failed_text}."
        )

    return dfs
