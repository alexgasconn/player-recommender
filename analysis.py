import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def _main_position(pos_value: str) -> str:
    text = str(pos_value or "").upper()
    if "GK" in text:
        return "GK"
    if "D" in text:
        return "D"
    if "M" in text:
        return "M"
    if "F" in text:
        return "F"
    return "UNK"


def _position_factor(base_position: str, other_position: str) -> float:
    if base_position == other_position:
        return 0.92
    if base_position == "M" and other_position in {"D", "F"}:
        return 0.98
    if base_position == "D" and other_position == "M":
        return 0.98
    if base_position == "F" and other_position == "M":
        return 0.98
    return 1.08


def _distance_to_similarity(distance_values: pd.Series) -> pd.Series:
    min_distance = distance_values.min()
    max_distance = distance_values.max()
    if pd.isna(min_distance) or pd.isna(max_distance) or max_distance == min_distance:
        return pd.Series([100.0] * len(distance_values), index=distance_values.index)
    return (1 - ((distance_values - min_distance) / (max_distance - min_distance))) * 100


def _safe_number(value, default=np.nan):
    if value is None:
        return default
    return pd.to_numeric(value, errors="coerce")


def _context_factor(
    row: pd.Series,
    target_comp,
    target_age,
    target_90s,
    quality_mode: str,
    prefer_same_comp: bool,
) -> float:
    comp_factor = 1.0
    if prefer_same_comp and target_comp:
        comp_factor = 0.96 if str(
            row.get("comp", "")) == str(target_comp) else 1.03

    age_factor = 1.0
    row_age = _safe_number(row.get("age_year", np.nan))
    if not pd.isna(target_age) and not pd.isna(row_age):
        age_gap_years = min(abs(float(row_age) - float(target_age)), 12.0)
        age_weight = {"strict": 0.12, "balanced": 0.06,
                      "exploratory": 0.02}.get(quality_mode, 0.06)
        age_factor = 1.0 + (age_gap_years / 12.0) * age_weight

    minutes_factor = 1.0
    row_90s = _safe_number(row.get("raw_90s", np.nan))
    if not pd.isna(target_90s) and not pd.isna(row_90s):
        max_90s = max(float(target_90s), float(row_90s), 1e-6)
        match_ratio = min(float(target_90s), float(row_90s)) / max_90s
        minutes_weight = {"strict": 0.25, "balanced": 0.12,
                          "exploratory": 0.04}.get(quality_mode, 0.12)
        minutes_factor = 1.0 + (1.0 - match_ratio) * minutes_weight

    return comp_factor * age_factor * minutes_factor


def find_closest_players_per_stat(
    dfs,
    player_name,
    position_weights,
    candidate_players=None,
    top_k=10,
    same_position_only=False,
    quality_mode="balanced",
    prefer_same_comp=False,
):
    closest_players_per_stat = {}
    combined_distances = {}
    combined_counts = {}

    first_df = next(iter(dfs.values()))
    if player_name not in first_df["player"].values:
        empty = pd.DataFrame(
            columns=[
                "player",
                "pos",
                "age",
                "squad",
                "comp",
                "similarity",
                "confidence",
                "groups_covered",
                "coverage_pct",
                "adjusted_distance",
            ]
        )
        return {}, empty

    base_row = first_df[first_df["player"] == player_name].iloc[0]
    player_position = _main_position(base_row.get("pos", "UNK"))
    target_comp = base_row.get("comp", "")
    target_age = _safe_number(base_row.get("age_year", np.nan))
    target_90s = _safe_number(base_row.get("raw_90s", np.nan))

    player_pool = set(first_df["player"].dropna().tolist())
    if candidate_players is not None:
        player_pool &= set(candidate_players)
    player_pool.add(player_name)

    for stat_name, df in dfs.items():
        if player_name not in df["player"].values:
            continue

        pca_columns = [
            col for col in df.columns if col.startswith(f"pca_{stat_name}_")]
        if not pca_columns:
            continue

        working_df = df[df["player"].isin(player_pool)].copy()
        if len(working_df) < 3:
            continue

        player_row = working_df[working_df["player"] == player_name]
        if player_row.empty:
            continue

        distances = cdist(player_row[pca_columns].iloc[[
                          0]], working_df[pca_columns], metric="euclidean").flatten()
        working_df["distance"] = distances

        weight = position_weights.get(stat_name, 1.0)
        working_df["weighted_distance"] = working_df["distance"] * weight
        working_df["position_factor"] = working_df["pos"].apply(
            lambda p: _position_factor(player_position, _main_position(p))
        )
        working_df["context_factor"] = working_df.apply(
            lambda row: _context_factor(
                row,
                target_comp=target_comp,
                target_age=target_age,
                target_90s=target_90s,
                quality_mode=quality_mode,
                prefer_same_comp=prefer_same_comp,
            ),
            axis=1,
        )
        working_df["final_distance"] = (
            working_df["weighted_distance"]
            * working_df["position_factor"]
            * working_df["context_factor"]
        )

        if same_position_only:
            working_df = working_df[working_df["pos"].apply(
                _main_position) == player_position]

        working_df = working_df[working_df["player"] != player_name].copy()
        if working_df.empty:
            continue

        working_df["similarity"] = _distance_to_similarity(
            working_df["final_distance"])
        stat_result = working_df.sort_values("final_distance").head(top_k)
        closest_players_per_stat[stat_name] = stat_result[
            ["player", "pos", "age", "squad", "comp",
                "final_distance", "similarity"]
        ].reset_index(drop=True)

        for _, row in working_df.iterrows():
            candidate = row["player"]
            combined_distances[candidate] = combined_distances.get(
                candidate, 0.0) + float(row["final_distance"])
            combined_counts[candidate] = combined_counts.get(candidate, 0) + 1

    if not combined_distances:
        empty = pd.DataFrame(
            columns=[
                "player",
                "pos",
                "age",
                "squad",
                "comp",
                "similarity",
                "confidence",
                "groups_covered",
                "coverage_pct",
                "adjusted_distance",
            ]
        )
        return closest_players_per_stat, empty

    total_groups_used = max(len(closest_players_per_stat), 1)
    combined_df = pd.DataFrame(
        [
            {
                "player": p,
                "avg_distance": combined_distances[p] / combined_counts[p],
                "groups_covered": combined_counts[p],
                "coverage_pct": (combined_counts[p] / total_groups_used) * 100,
            }
            for p in combined_distances
        ]
    )

    combined_df["coverage_factor"] = 1 + \
        (1 - combined_df["coverage_pct"] / 100) * 0.2
    combined_df["adjusted_distance"] = combined_df["avg_distance"] * \
        combined_df["coverage_factor"]

    merge_cols = ["player", "pos", "age", "squad"]
    if "comp" in first_df.columns:
        merge_cols.append("comp")
    if "raw_90s" in first_df.columns:
        merge_cols.append("raw_90s")

    combined_df = combined_df.merge(
        first_df[merge_cols].drop_duplicates(subset=["player"]),
        on="player",
        how="left",
    )

    if not pd.isna(target_90s) and "raw_90s" in combined_df.columns:
        safe_target_90s = max(float(target_90s), 1e-6)
        combined_df["minutes_match"] = combined_df["raw_90s"].apply(
            lambda x: min(float(x), safe_target_90s) /
            max(float(x), safe_target_90s)
            if pd.notna(x)
            else 0.0
        )
    else:
        combined_df["minutes_match"] = 0.5

    combined_df["similarity"] = _distance_to_similarity(
        combined_df["adjusted_distance"])
    combined_df["confidence"] = (
        (combined_df["coverage_pct"] * 0.6) +
        (combined_df["minutes_match"] * 100 * 0.4)
    ).round(1)

    combined_df = combined_df.sort_values(
        "adjusted_distance").head(top_k).copy()
    combined_df["similarity"] = combined_df["similarity"].round(2)

    return closest_players_per_stat, combined_df[
        [
            "player",
            "pos",
            "age",
            "squad",
            "comp",
            "similarity",
            "confidence",
            "groups_covered",
            "coverage_pct",
            "adjusted_distance",
        ]
    ]


def compute_stat_group_means(dfs, player_name, stat_group=None, reference_players=None):
    stat_group_means = {}
    stat_names = [stat_group] if stat_group else list(dfs.keys())

    for stat_name in stat_names:
        if stat_name not in dfs:
            continue
        df = dfs[stat_name]

        if reference_players is not None:
            df = df[df["player"].isin(reference_players)].copy()
            if df.empty:
                continue

        if player_name not in df["player"].values:
            continue

        numerical_cols = [
            col
            for col in df.select_dtypes(include=np.number).columns
            if not col.startswith("pca_") and not col.startswith("cluster_")
        ]
        if not numerical_cols:
            continue

        percentile_table = df[numerical_cols].rank(pct=True) * 100
        player_idx = df.index[df["player"] == player_name]
        if len(player_idx) == 0:
            continue

        player_percentiles = percentile_table.loc[player_idx[0]]
        stat_group_means[stat_name] = round(
            float(player_percentiles.mean()), 2)

    return stat_group_means


def get_position_weights(selected_player, dfs):
    first_df = next(iter(dfs.values()))
    player_position = _main_position(
        first_df[first_df["player"] == selected_player]["pos"].values[0])

    weights = {
        "standard_stats": 1.0,
        "shooting": 1.0,
        "passing": 1.0,
        "pass_types": 1.0,
        "goal_and_shot_creation": 1.0,
        "defensive_actions": 1.0,
        "possession": 1.0,
        "playing_time": 1.0,
        "miscellaneous_stats": 1.0,
    }

    if player_position == "F":
        weights.update(
            {
                "shooting": 1.5,
                "goal_and_shot_creation": 1.5,
                "passing": 1.2,
                "defensive_actions": 0.5,
            }
        )
    elif player_position == "M":
        weights.update(
            {
                "passing": 1.5,
                "possession": 1.5,
                "goal_and_shot_creation": 1.3,
                "shooting": 1.0,
                "defensive_actions": 1.0,
            }
        )
    elif player_position == "D":
        weights.update(
            {
                "defensive_actions": 1.5,
                "possession": 1.3,
                "playing_time": 1.2,
                "shooting": 0.5,
                "goal_and_shot_creation": 0.5,
            }
        )

    return weights
