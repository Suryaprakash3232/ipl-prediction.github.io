"""
IPL Match Prediction System - Prediction Engine
=================================================
Loads trained models and provides an easy-to-use interface for making
match predictions with win probabilities and player performance insights.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Optional

from config import (
    MODEL_DIR, MATCHES_CSV, DELIVERIES_CSV,
    CURRENT_TEAMS, VENUES, TEAM_NAME_MAP
)
from feature_engineering import (
    normalize_team_name, load_and_clean_data,
    compute_rolling_win_rate, compute_team_win_rates,
    compute_h2h_record, compute_venue_win_rate, compute_venue_avg_score,
    compute_team_batting_stats, compute_team_bowling_stats,
    get_top_batsman, get_top_bowler,
    FEATURE_COLUMNS, H2H_MIN_MATCHES
)


class IPLPredictor:
    """
    IPL Match Outcome Predictor.

    Loads a trained model and historical data, then predicts match outcomes
    with win probabilities and optional player performance analysis.

    Usage:
        predictor = IPLPredictor()
        result = predictor.predict(
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            venue="Wankhede Stadium, Mumbai",
            toss_winner="Mumbai Indians",
            toss_decision="bat"
        )
        predictor.display_prediction(result)
    """

    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize the predictor by loading model and historical data.

        Args:
            model_path: Path to saved model pickle. Defaults to models/best_model.pkl
            scaler_path: Path to saved scaler pickle. Defaults to models/scaler.pkl
        """
        model_path = model_path or os.path.join(MODEL_DIR, "best_model.pkl")
        scaler_path = scaler_path or os.path.join(MODEL_DIR, "scaler.pkl")
        meta_path = os.path.join(MODEL_DIR, "model_meta.pkl")

        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run main.py first to train the model."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        self.meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        self.use_scaled = self.meta.get("use_scaled", False)

        # Load historical data
        self.matches, self.deliveries = load_and_clean_data(MATCHES_CSV, DELIVERIES_CSV)

        print(f"✅ IPL Predictor loaded successfully!")
        print(f"   Model: {self.meta.get('model_name', 'Unknown')}")
        print(f"   Training metrics: {self.meta.get('metrics', {})}")
        print(f"   Historical data: {len(self.matches)} matches loaded\n")

    def _validate_inputs(self, team1: str, team2: str, venue: str,
                          toss_winner: str, toss_decision: str):
        """Validate prediction inputs."""
        team1 = normalize_team_name(team1)
        team2 = normalize_team_name(team2)
        toss_winner = normalize_team_name(toss_winner)

        # Check teams exist in data
        all_teams = set(self.matches["team1"].tolist() + self.matches["team2"].tolist())
        for team, label in [(team1, "Team 1"), (team2, "Team 2")]:
            if team not in all_teams:
                raise ValueError(
                    f"{label} '{team}' not found in historical data.\n"
                    f"Available teams: {sorted(all_teams)}"
                )

        if team1 == team2:
            raise ValueError("Team 1 and Team 2 must be different teams.")

        if toss_winner not in [team1, team2]:
            raise ValueError(
                f"Toss winner '{toss_winner}' must be either '{team1}' or '{team2}'."
            )

        if toss_decision.lower() not in ["bat", "field"]:
            raise ValueError("Toss decision must be 'bat' or 'field'.")

        return team1, team2, venue, toss_winner, toss_decision.lower()

    def _build_prediction_features(self, team1: str, team2: str, venue: str,
                                     toss_winner: str, toss_decision: str) -> np.ndarray:
        """Build feature vector for a single prediction."""
        # Use latest date as reference
        latest_date = self.matches["date"].max() + pd.Timedelta(days=1)

        # Team form
        t1_form = compute_rolling_win_rate(self.matches, team1, latest_date)
        t2_form = compute_rolling_win_rate(self.matches, team2, latest_date)

        # Overall win rates
        win_rates = compute_team_win_rates(self.matches)
        t1_overall = win_rates.get(team1, 0.5)
        t2_overall = win_rates.get(team2, 0.5)

        # Head-to-head
        h2h = compute_h2h_record(self.matches, team1, team2)
        h2h_pct = h2h["team1_win_pct"] if h2h["total"] >= H2H_MIN_MATCHES else 0.5

        # Venue stats
        t1_venue = compute_venue_win_rate(self.matches, team1, venue)
        t2_venue = compute_venue_win_rate(self.matches, team2, venue)
        venue_avg = compute_venue_avg_score(self.deliveries, self.matches, venue)

        # Batting & bowling
        t1_bat = compute_team_batting_stats(self.deliveries, self.matches, team1)
        t2_bat = compute_team_batting_stats(self.deliveries, self.matches, team2)
        t1_bowl = compute_team_bowling_stats(self.deliveries, self.matches, team1)
        t2_bowl = compute_team_bowling_stats(self.deliveries, self.matches, team2)

        # Toss
        toss_won_by_team1 = 1 if toss_winner == team1 else 0
        chose_bat = 1 if toss_decision == "bat" else 0

        features = np.array([
            t1_form, t2_form, t1_form - t2_form,
            t1_overall, t2_overall, t1_overall - t2_overall,
            h2h_pct, h2h["total"],
            t1_venue, t2_venue, venue_avg,
            t1_bat["avg_score"], t2_bat["avg_score"],
            t1_bat["avg_sr"], t2_bat["avg_sr"],
            t1_bat["avg_score"] - t2_bat["avg_score"],
            t1_bowl["avg_conceded"], t2_bowl["avg_conceded"],
            t1_bowl["avg_wickets"], t2_bowl["avg_wickets"],
            t1_bowl["economy"] - t2_bowl["economy"],
            toss_won_by_team1, chose_bat,
        ]).reshape(1, -1)

        # Handle NaN/inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    def predict(self, team1: str, team2: str, venue: str,
                toss_winner: str, toss_decision: str,
                include_players: bool = True,
                squad1: Optional[list] = None,
                squad2: Optional[list] = None) -> dict:
        """
        Predict the outcome of an IPL match.

        Args:
            team1: Name of the first team.
            team2: Name of the second team.
            venue: Match venue.
            toss_winner: Team that won the toss.
            toss_decision: Toss decision ('bat' or 'field').
            include_players: If True, include player performance predictions.

        Returns:
            dict with prediction results including:
              - predicted_winner: Name of predicted winning team
              - team1_win_prob: Win probability for team1 (%)
              - team2_win_prob: Win probability for team2 (%)
              - confidence: Prediction confidence level
              - h2h_record: Head-to-head stats
              - player_analysis: (optional) Top players prediction
        """
        # Validate inputs
        team1, team2, venue, toss_winner, toss_decision = self._validate_inputs(
            team1, team2, venue, toss_winner, toss_decision
        )

        # Build features
        features = self._build_prediction_features(
            team1, team2, venue, toss_winner, toss_decision
        )

        # Scale if needed
        if self.use_scaled and self.scaler:
            features = self.scaler.transform(features)

        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        team1_prob = probabilities[1] * 100
        team2_prob = probabilities[0] * 100
        predicted_winner = team1 if prediction == 1 else team2
        win_prob = max(team1_prob, team2_prob)

        # Confidence level
        if win_prob >= 70:
            confidence = "High"
        elif win_prob >= 60:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Head-to-head
        h2h = compute_h2h_record(self.matches, team1, team2)

        result = {
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "predicted_winner": predicted_winner,
            "team1_win_prob": round(team1_prob, 2),
            "team2_win_prob": round(team2_prob, 2),
            "confidence": confidence,
            "h2h_record": {
                "total_matches": h2h["total"],
                f"{team1}_wins": h2h["team1_wins"],
                f"{team2}_wins": h2h["team2_wins"],
            },
        }

        # Player performance analysis
        if include_players:
            analysis = self._get_player_analysis(team1, team2, squad1, squad2)
            result["player_analysis"] = analysis
            
            # --- Scoreboard Prediction ---
            team1_bats_first = (toss_winner == team1 and toss_decision == "bat") or (toss_winner == team2 and toss_decision == "field")

            t1_bat = compute_team_batting_stats(self.deliveries, self.matches, team1)
            t2_bat = compute_team_batting_stats(self.deliveries, self.matches, team2)

            if team1_bats_first:
                inn1_team, inn2_team = team1, team2
                inn1_score = int(t1_bat["avg_score"])
                inn2_score = int(t2_bat["avg_score"])
            else:
                inn1_team, inn2_team = team2, team1
                inn1_score = int(t2_bat["avg_score"])
                inn2_score = int(t1_bat["avg_score"])

            # Force scores to align with predicted winner
            if predicted_winner == inn1_team and inn1_score <= inn2_score:
                inn1_score = inn2_score + np.random.randint(5, 20)
            elif predicted_winner == inn2_team and inn2_score <= inn1_score:
                inn2_score = inn1_score + 1
            elif predicted_winner == inn1_team and inn2_score >= inn1_score:
                inn1_score = inn2_score + np.random.randint(5, 20)

            inn1_wickets = np.random.randint(4, 9)
            inn2_wickets = np.random.randint(4, 10) if predicted_winner == inn1_team else np.random.randint(2, 7)

            # --- Simulated Player Performances in Scorecard ---
            # Innings 1 Top Bat (from inn1_team)
            t1_scorer = analysis[inn1_team]["likely_top_scorer"]
            t1_scorer_avg = t1_scorer["recent_avg_runs"] or 30.0
            t1_runs = int(np.random.normal(t1_scorer_avg, 12))
            t1_runs = min(inn1_score - 15, max(12, t1_runs))
            t1_balls = int(t1_runs / np.random.uniform(1.2, 1.7))
            inn1_top_bat_str = f"{t1_scorer['name']} {t1_runs}({t1_balls})"

            # Innings 1 Top Bowl (from inn2_team)
            t2_bowler = analysis[inn2_team]["likely_top_wicket_taker"]
            t2_bowler_avg = t2_bowler["recent_avg_wickets"] or 2.0
            t2_wkts = min(inn1_wickets, max(1, int(np.random.normal(t2_bowler_avg, 0.7))))
            t2_runs_conceded = np.random.randint(15, 42)
            inn1_top_bowl_str = f"{t2_bowler['name']} {t2_wkts}/{t2_runs_conceded}"

            # Innings 2 Top Bat (from inn2_team)
            t2_scorer = analysis[inn2_team]["likely_top_scorer"]
            t2_scorer_avg = t2_scorer["recent_avg_runs"] or 30.0
            t2_runs = int(np.random.normal(t2_scorer_avg, 12))
            t2_runs = min(inn2_score - 15, max(12, t2_runs))
            t2_balls = int(t2_runs / np.random.uniform(1.2, 1.7))
            inn2_top_bat_str = f"{t2_scorer['name']} {t2_runs}({t2_balls})"

            # Innings 2 Top Bowl (from inn1_team)
            t1_bowler = analysis[inn1_team]["likely_top_wicket_taker"]
            t1_bowler_avg = t1_bowler["recent_avg_wickets"] or 2.0
            t1_wkts = min(inn2_wickets, max(1, int(np.random.normal(t1_bowler_avg, 0.7))))
            t1_runs_conceded = np.random.randint(15, 42)
            inn2_top_bowl_str = f"{t1_bowler['name']} {t1_wkts}/{t1_runs_conceded}"

            result["scoreboard"] = {
                "inn1": {
                    "team": inn1_team,
                    "runs": inn1_score,
                    "wickets": inn1_wickets,
                    "top_bat": inn1_top_bat_str,
                    "top_bowl": inn1_top_bowl_str
                },
                "inn2": {
                    "team": inn2_team,
                    "runs": inn2_score,
                    "wickets": inn2_wickets,
                    "top_bat": inn2_top_bat_str,
                    "top_bowl": inn2_top_bowl_str
                }
            }

            # --- Player of the Match Prediction ---
            win_analysis = analysis[predicted_winner]
            if np.random.random() > 0.4:
                result["potm"] = win_analysis["likely_top_scorer"]["name"]
            else:
                result["potm"] = win_analysis["likely_top_wicket_taker"]["name"]

        return result

    def _get_player_analysis(self, team1: str, team2: str, squad1: Optional[list] = None, squad2: Optional[list] = None) -> dict:
        """Get top performer predictions for both teams, filtering by selected squads if available."""
        analysis = {}

        s1_set = set(squad1) if squad1 else None
        s2_set = set(squad2) if squad2 else None

        for team, squad in [(team1, s1_set), (team2, s2_set)]:
            top_bat = get_top_batsman(self.deliveries, self.matches, team, allowed_players=squad)
            top_bowl = get_top_bowler(self.deliveries, self.matches, team, allowed_players=squad)

            analysis[team] = {
                "likely_top_scorer": {
                    "name": top_bat["name"],
                    "recent_avg_runs": top_bat["avg_runs"],
                },
                "likely_top_wicket_taker": {
                    "name": top_bowl["name"],
                    "recent_avg_wickets": top_bowl["avg_wickets"],
                },
            }

        return analysis

    def display_prediction(self, result: dict):
        """Display prediction results in a formatted output."""
        print("\n" + "═" * 65)
        print("  🏏  IPL MATCH PREDICTION  🏏")
        print("═" * 65)

        print(f"\n  📍 Venue: {result['venue']}")
        print(f"  🪙 Toss: {result['toss_winner']} won → chose to {result['toss_decision']}")

        print(f"\n  {'─' * 55}")
        print(f"  {'TEAM':<30} {'WIN PROBABILITY':>20}")
        print(f"  {'─' * 55}")

        # Team 1
        t1_bar = "█" * int(result["team1_win_prob"] / 5)
        t1_marker = " 🏆" if result["predicted_winner"] == result["team1"] else ""
        print(f"  {result['team1']:<30} {result['team1_win_prob']:>6.1f}% {t1_bar}{t1_marker}")

        # Team 2
        t2_bar = "█" * int(result["team2_win_prob"] / 5)
        t2_marker = " 🏆" if result["predicted_winner"] == result["team2"] else ""
        print(f"  {result['team2']:<30} {result['team2_win_prob']:>6.1f}% {t2_bar}{t2_marker}")

        print(f"\n  🎯 PREDICTED WINNER: {result['predicted_winner']}")
        print(f"  📊 Confidence: {result['confidence']}")

        # H2H
        h2h = result["h2h_record"]
        print(f"\n  📜 Head-to-Head ({h2h['total_matches']} matches):")
        for key, val in h2h.items():
            if key != "total_matches":
                print(f"     {key}: {val}")

        # Player analysis
        if "player_analysis" in result:
            print(f"\n  👤 Player Performance Predictions:")
            print(f"  {'─' * 55}")
            for team, data in result["player_analysis"].items():
                ts = data["likely_top_scorer"]
                tw = data["likely_top_wicket_taker"]
                print(f"  {team}:")
                print(f"     🏏 Likely Top Scorer:       {ts['name']} (avg {ts['recent_avg_runs']} runs)")
                print(f"     🎳 Likely Top Wicket-Taker: {tw['name']} (avg {tw['recent_avg_wickets']} wkts)")

        print("\n" + "═" * 65)

    def predict_multiple(self, matches_list: list) -> list:
        """
        Predict outcomes for multiple matches.

        Args:
            matches_list: List of dicts, each with keys:
                team1, team2, venue, toss_winner, toss_decision

        Returns:
            List of prediction result dicts.
        """
        results = []
        for i, match in enumerate(matches_list, 1):
            print(f"\n--- Match {i}/{len(matches_list)} ---")
            try:
                result = self.predict(**match)
                results.append(result)
            except (ValueError, KeyError) as e:
                print(f"   ⚠️  Skipped: {e}")
                results.append({"error": str(e)})
        return results

    @staticmethod
    def list_teams():
        """Print all available teams."""
        print("\n🏏 Available IPL Teams:")
        print("─" * 35)
        for i, team in enumerate(CURRENT_TEAMS, 1):
            print(f"  {i:2d}. {team}")

    @staticmethod
    def list_venues():
        """Print all available venues."""
        print("\n🏟️  Available Venues:")
        print("─" * 50)
        for i, venue in enumerate(VENUES, 1):
            print(f"  {i:2d}. {venue}")


def quick_predict(team1: str, team2: str, venue: str,
                   toss_winner: str, toss_decision: str) -> dict:
    """
    Convenience function for quick single-match predictions.

    Usage:
        result = quick_predict(
            "Mumbai Indians", "Chennai Super Kings",
            "Wankhede Stadium, Mumbai",
            "Mumbai Indians", "bat"
        )
    """
    predictor = IPLPredictor()
    result = predictor.predict(team1, team2, venue, toss_winner, toss_decision)
    predictor.display_prediction(result)
    return result
