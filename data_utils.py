# data_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
from config import urls, stat_keys

@st.cache_resource
def load_and_process(url_key):
    df = pd.read_html(urls[url_key])[0]
    df.columns = df.columns.droplevel()
    df.columns = df.columns.str.strip().str.lower()

    categorical_cols = ['player', 'nation', 'pos', 'squad', 'comp', 'age', 'born', 'matches']
    for col in df.columns:
        if col not in categorical_cols and isinstance(df[col], pd.Series):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(df.median(numeric_only=True), inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df['pos'] = df['pos'].str.split(',').str[0]

    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

@st.cache_resource
def apply_pca_kmeans(df: pd.DataFrame, name: str):
    df_numerical = df.select_dtypes(include=np.number)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_numerical)
    df[f'pca1_{name}'] = df_pca[:, 0]
    df[f'pca2_{name}'] = df_pca[:, 1]

    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(df_pca)
    df[f'cluster_{name}'] = kmeans.labels_

    return df

@st.cache_resource
def load_all_stat_dfs():
    dfs = {}
    for key in stat_keys:
        df = load_and_process(key)
        df = apply_pca_kmeans(df, key)
        dfs[key] = df
    return dfs