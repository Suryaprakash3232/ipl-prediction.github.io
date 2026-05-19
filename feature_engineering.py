"""
IPL Match Prediction System - Feature Engineering
===================================================
Transforms raw match and delivery data into ML-ready features.

Feature categories:
  1. Team Performance: Win rate, recent form, scoring averages
  2. Head-to-Head: Historical matchup records
  3. Venue Impact: Team performance at specific grounds
  4. Toss Influence: Toss win/decision correlations
  5. Player Performance: Aggregated batting/bowling stats
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from config import TEAM_NAME_MAP, RECENT_FORM_WINDOW, H2H_MIN_MATCHES


def normalize_team_name(name: str) -> str:
    """Normalize historical team names to current names."""
    return TEAM_NAME_MAP.get(name, name)


def load_and_clean_data(matches_path: str, deliveries_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSV files and perform initial cleaning.

    Returns:
        Tuple of (matches_df, deliveries_df) with cleaned data.
    """
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    # Normalize team names
    for col in ["team1", "team2", "winner", "toss_winner"]:
        if col in matches.columns:
            matches[col] = matches[col].apply(normalize_team_name)

    for col in ["batting_team", "bowling_team"]:
        if col in deliveries.columns:
            deliveries[col] = deliveries[col].apply(normalize_team_name)

    # Parse dates
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    # Drop matches with no result
    matches = matches.dropna(subset=["winner"])

    return matches, deliveries


# ──────────────────────────────────────────────
# 1. Team Overall Win Rate
# ──────────────────────────────────────────────

def compute_team_win_rates(matches: pd.DataFrame) -> dict:
    """
    Compute overall win rate for each team up to each match (expanding window).

    Returns:
        dict mapping team -> cumulative win rate.
    """
    team_stats = {}
    for team in set(matches["team1"].tolist() + matches["team2"].tolist()):
        team_matches = matches[(matches["team1"] == team) | (matches["team2"] == team)]
        total = len(team_matches)
        wins = len(team_matches[team_matches["winner"] == team])
        team_stats[team] = wins / total if total > 0 else 0.5
    return team_stats


def compute_rolling_win_rate(matches: pd.DataFrame, team: str, before_date: pd.Timestamp,
                              window: int = RECENT_FORM_WINDOW) -> float:
    """Compute win rate for a team over the last `window` matches before a given date."""
    team_matches = matches[
        ((matches["team1"] == team) | (matches["team2"] == team)) &
        (matches["date"] < before_date)
    ].tail(window)

    if len(team_matches) == 0:
        return 0.5

    wins = len(team_matches[team_matches["winner"] == team])
    return wins / len(team_matches)


# ──────────────────────────────────────────────
# 2. Head-to-Head Record
# ──────────────────────────────────────────────

def compute_h2h_record(matches: pd.DataFrame, team1: str, team2: str,
                        before_date: Optional[pd.Timestamp] = None) -> dict:
    """Compute head-to-head record between two teams."""
    mask = (
        ((matches["team1"] == team1) & (matches["team2"] == team2)) |
        ((matches["team1"] == team2) & (matches["team2"] == team1))
    )
    if before_date is not None:
        mask &= matches["date"] < before_date

    h2h = matches[mask]
    total = len(h2h)

    if total == 0:
        return {"team1_wins": 0, "team2_wins": 0, "total": 0, "team1_win_pct": 0.5}

    t1_wins = len(h2h[h2h["winner"] == team1])
    t2_wins = len(h2h[h2h["winner"] == team2])

    return {
        "team1_wins": t1_wins,
        "team2_wins": t2_wins,
        "total": total,
        "team1_win_pct": t1_wins / total,
    }


# ──────────────────────────────────────────────
# 3. Venue Statistics
# ──────────────────────────────────────────────

def compute_venue_win_rate(matches: pd.DataFrame, team: str, venue: str,
                            before_date: Optional[pd.Timestamp] = None) -> float:
    """Win rate for a team at a specific venue."""
    mask = (
        ((matches["team1"] == team) | (matches["team2"] == team)) &
        (matches["venue"] == venue)
    )
    if before_date is not None:
        mask &= matches["date"] < before_date

    venue_matches = matches[mask]
    if len(venue_matches) == 0:
        return 0.5

    wins = len(venue_matches[venue_matches["winner"] == team])
    return wins / len(venue_matches)


def compute_venue_avg_score(deliveries: pd.DataFrame, matches: pd.DataFrame,
                             venue: str, before_date: Optional[pd.Timestamp] = None) -> float:
    """Average first-innings score at a venue."""
    venue_matches = matches[matches["venue"] == venue]
    if before_date is not None:
        venue_matches = venue_matches[venue_matches["date"] < before_date]

    if len(venue_matches) == 0:
        return 160.0  # Default average

    match_ids = venue_matches["match_id"].tolist()
    first_inning = deliveries[
        (deliveries["match_id"].isin(match_ids)) & (deliveries["inning"] == 1)
    ]

    if len(first_inning) == 0:
        return 160.0

    scores = first_inning.groupby("match_id")["total_runs"].sum()
    return scores.mean()


# ──────────────────────────────────────────────
# 4. Batting & Bowling Aggregates
# ──────────────────────────────────────────────

def compute_team_batting_stats(deliveries: pd.DataFrame, matches: pd.DataFrame,
                                team: str, before_date: Optional[pd.Timestamp] = None) -> dict:
    """Aggregate batting stats for a team."""
    team_match_ids = matches[
        ((matches["team1"] == team) | (matches["team2"] == team))
    ]
    if before_date is not None:
        team_match_ids = team_match_ids[team_match_ids["date"] < before_date]

    team_match_ids = team_match_ids["match_id"].tolist()

    batting = deliveries[
        (deliveries["match_id"].isin(team_match_ids)) &
        (deliveries["batting_team"] == team)
    ]

    if len(batting) == 0:
        return {"avg_score": 160.0, "avg_sr": 130.0, "avg_boundaries": 15.0}

    per_match = batting.groupby("match_id").agg(
        total=("total_runs", "sum"),
        balls=("ball", "count"),
        fours=("batsman_runs", lambda x: (x == 4).sum()),
        sixes=("batsman_runs", lambda x: (x == 6).sum()),
    )

    return {
        "avg_score": per_match["total"].mean(),
        "avg_sr": (per_match["total"].sum() / per_match["balls"].sum()) * 100 if per_match["balls"].sum() > 0 else 130.0,
        "avg_boundaries": (per_match["fours"] + per_match["sixes"]).mean(),
    }


def compute_team_bowling_stats(deliveries: pd.DataFrame, matches: pd.DataFrame,
                                team: str, before_date: Optional[pd.Timestamp] = None) -> dict:
    """Aggregate bowling stats for a team."""
    team_match_ids = matches[
        ((matches["team1"] == team) | (matches["team2"] == team))
    ]
    if before_date is not None:
        team_match_ids = team_match_ids[team_match_ids["date"] < before_date]

    team_match_ids = team_match_ids["match_id"].tolist()

    bowling = deliveries[
        (deliveries["match_id"].isin(team_match_ids)) &
        (deliveries["bowling_team"] == team)
    ]

    if len(bowling) == 0:
        return {"avg_conceded": 160.0, "avg_wickets": 6.0, "economy": 8.0}

    per_match = bowling.groupby("match_id").agg(
        conceded=("total_runs", "sum"),
        wickets=("is_wicket", "sum"),
        balls=("ball", "count"),
    )

    return {
        "avg_conceded": per_match["conceded"].mean(),
        "avg_wickets": per_match["wickets"].mean(),
        "economy": (per_match["conceded"].sum() / (per_match["balls"].sum() / 6)) if per_match["balls"].sum() > 0 else 8.0,
    }


# ──────────────────────────────────────────────
# 5. Top Player Performance Features
# ──────────────────────────────────────────────

def get_top_batsman(deliveries: pd.DataFrame, matches: pd.DataFrame,
                     team: str, n_recent: int = 10,
                     before_date: Optional[pd.Timestamp] = None) -> dict:
    """Find the likely top scorer for a team based on recent form."""
    recent_match_ids = matches[
        ((matches["team1"] == team) | (matches["team2"] == team))
    ]
    if before_date is not None:
        recent_match_ids = recent_match_ids[recent_match_ids["date"] < before_date]

    recent_match_ids = recent_match_ids.tail(n_recent)["match_id"].tolist()

    batting = deliveries[
        (deliveries["match_id"].isin(recent_match_ids)) &
        (deliveries["batting_team"] == team)
    ]

    if len(batting) == 0:
        return {"name": "Unknown", "avg_runs": 0, "matches": 0}

    player_stats = batting.groupby("batsman").agg(
        total_runs=("batsman_runs", "sum"),
        matches=("match_id", "nunique"),
    )
    player_stats["avg_runs"] = player_stats["total_runs"] / player_stats["matches"]

    top = player_stats.sort_values("total_runs", ascending=False).iloc[0]
    return {
        "name": top.name,
        "avg_runs": round(top["avg_runs"], 1),
        "matches": int(top["matches"]),
    }


def get_top_bowler(deliveries: pd.DataFrame, matches: pd.DataFrame,
                    team: str, n_recent: int = 10,
                    before_date: Optional[pd.Timestamp] = None) -> dict:
    """Find the likely leading wicket-taker for a team based on recent form."""
    recent_match_ids = matches[
        ((matches["team1"] == team) | (matches["team2"] == team))
    ]
    if before_date is not None:
        recent_match_ids = recent_match_ids[recent_match_ids["date"] < before_date]

    recent_match_ids = recent_match_ids.tail(n_recent)["match_id"].tolist()

    bowling = deliveries[
        (deliveries["match_id"].isin(recent_match_ids)) &
        (deliveries["bowling_team"] == team)
    ]

    if len(bowling) == 0:
        return {"name": "Unknown", "avg_wickets": 0, "matches": 0}

    player_stats = bowling.groupby("bowler").agg(
        total_wickets=("is_wicket", "sum"),
        matches=("match_id", "nunique"),
    )
    player_stats["avg_wickets"] = player_stats["total_wickets"] / player_stats["matches"]

    top = player_stats.sort_values("total_wickets", ascending=False).iloc[0]
    return {
        "name": top.name,
        "avg_wickets": round(top["avg_wickets"], 2),
        "matches": int(top["matches"]),
    }


# ──────────────────────────────────────────────
# Master Feature Builder
# ──────────────────────────────────────────────

def build_features(matches: pd.DataFrame, deliveries: pd.DataFrame,
                    use_expanding_window: bool = True) -> pd.DataFrame:
    """
    Build the full feature matrix for all matches.

    Each row corresponds to one match with features computed from
    data available *before* that match (to avoid data leakage).

    Args:
        matches: Cleaned matches DataFrame.
        deliveries: Cleaned deliveries DataFrame.
        use_expanding_window: If True, only use data before each match date.

    Returns:
        pd.DataFrame with feature columns and target variable.
    """
    print("🔧 Building feature matrix...")
    features_list = []

    total = len(matches)
    for idx, row in matches.iterrows():
        if idx % 100 == 0:
            print(f"   Processing match {idx + 1}/{total}...")

        team1, team2 = row["team1"], row["team2"]
        venue = row["venue"]
        match_date = row["date"]
        before = match_date if use_expanding_window else None

        # --- Team form ---
        t1_form = compute_rolling_win_rate(matches, team1, match_date)
        t2_form = compute_rolling_win_rate(matches, team2, match_date)

        # --- Overall win rates ---
        early_matches = matches[matches["date"] < match_date]
        win_rates = compute_team_win_rates(early_matches) if len(early_matches) > 0 else {}
        t1_overall = win_rates.get(team1, 0.5)
        t2_overall = win_rates.get(team2, 0.5)

        # --- Head-to-head ---
        h2h = compute_h2h_record(matches, team1, team2, before)
        h2h_pct = h2h["team1_win_pct"] if h2h["total"] >= H2H_MIN_MATCHES else 0.5

        # --- Venue stats ---
        t1_venue = compute_venue_win_rate(matches, team1, venue, before)
        t2_venue = compute_venue_win_rate(matches, team2, venue, before)
        venue_avg = compute_venue_avg_score(deliveries, matches, venue, before)

        # --- Batting/Bowling ---
        t1_bat = compute_team_batting_stats(deliveries, matches, team1, before)
        t2_bat = compute_team_batting_stats(deliveries, matches, team2, before)
        t1_bowl = compute_team_bowling_stats(deliveries, matches, team1, before)
        t2_bowl = compute_team_bowling_stats(deliveries, matches, team2, before)

        # --- Toss ---
        toss_won_by_team1 = 1 if row["toss_winner"] == team1 else 0
        chose_bat = 1 if row["toss_decision"] == "bat" else 0

        # --- Target: Team 1 wins ---
        target = 1 if row["winner"] == team1 else 0

        features_list.append({
            "match_id": row["match_id"],
            "season": row["season"],
            "team1": team1,
            "team2": team2,
            "venue": venue,
            # Form features
            "team1_recent_form": t1_form,
            "team2_recent_form": t2_form,
            "form_diff": t1_form - t2_form,
            # Overall
            "team1_overall_wr": t1_overall,
            "team2_overall_wr": t2_overall,
            "overall_wr_diff": t1_overall - t2_overall,
            # H2H
            "h2h_team1_pct": h2h_pct,
            "h2h_matches": h2h["total"],
            # Venue
            "team1_venue_wr": t1_venue,
            "team2_venue_wr": t2_venue,
            "venue_avg_score": venue_avg,
            # Batting
            "team1_avg_score": t1_bat["avg_score"],
            "team2_avg_score": t2_bat["avg_score"],
            "team1_avg_sr": t1_bat["avg_sr"],
            "team2_avg_sr": t2_bat["avg_sr"],
            "batting_score_diff": t1_bat["avg_score"] - t2_bat["avg_score"],
            # Bowling
            "team1_avg_conceded": t1_bowl["avg_conceded"],
            "team2_avg_conceded": t2_bowl["avg_conceded"],
            "team1_avg_wickets": t1_bowl["avg_wickets"],
            "team2_avg_wickets": t2_bowl["avg_wickets"],
            "bowling_economy_diff": t1_bowl["economy"] - t2_bowl["economy"],
            # Toss
            "toss_won_by_team1": toss_won_by_team1,
            "chose_bat": chose_bat,
            # Meta
            "target": target,
        })

    features_df = pd.DataFrame(features_list)
    print(f"   ✅ Feature matrix built: {features_df.shape}")
    return features_df


# All numeric feature column names used by the model
FEATURE_COLUMNS = [
    "team1_recent_form", "team2_recent_form", "form_diff",
    "team1_overall_wr", "team2_overall_wr", "overall_wr_diff",
    "h2h_team1_pct", "h2h_matches",
    "team1_venue_wr", "team2_venue_wr", "venue_avg_score",
    "team1_avg_score", "team2_avg_score",
    "team1_avg_sr", "team2_avg_sr",
    "batting_score_diff",
    "team1_avg_conceded", "team2_avg_conceded",
    "team1_avg_wickets", "team2_avg_wickets",
    "bowling_economy_diff",
    "toss_won_by_team1", "chose_bat",
]
