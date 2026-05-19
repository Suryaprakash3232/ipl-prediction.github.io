"""
IPL Match Prediction System - Data Generator
==============================================
Generates realistic synthetic IPL match data for training and testing.
This module creates matches.csv, deliveries.csv, and players.csv with
statistically plausible distributions based on real IPL patterns.

If you have real IPL data from Kaggle, you can skip this module entirely
and place your CSV files in the data/ directory.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import (
    DATA_DIR, CURRENT_TEAMS, HISTORICAL_TEAMS, VENUES,
    SEASON_START_YEAR, RANDOM_STATE
)

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# ──────────────────────────────────────────────
# Player database (representative archetypes)
# ──────────────────────────────────────────────
# IPL 2026 Squads — Updated post December 2025 mini-auction
PLAYER_ARCHETYPES = {
    # Captain: Hardik Pandya | Key additions: Trent Boult, Reece Topley
    "Mumbai Indians": {
        "batsmen": ["Rohit Sharma", "Ishan Kishan", "Suryakumar Yadav", "Tilak Varma", "Naman Dhir"],
        "bowlers": ["Jasprit Bumrah", "Gerald Coetzee", "Akash Madhwal", "Nuwan Thushara", "Piyush Chawla"],
        "allrounders": ["Hardik Pandya","namin thir"],
    },
    "Chennai Super Kings": {
        "batsmen": ["Ruturaj Gaikwad", "Rachin Ravindra", "MS Dhoni", "Shaik Rasheed", "Anshul Kamboj"],
        "bowlers": ["Matheesha Pathirana", "Khaleel Ahmed", "Noor Ahmad", "Anshul Kamboj", "Gurjapneet Singh"],
        "allrounders": ["Ravindra Jadeja", "Shivam Dube", "Vijay Shankar"],
    },
    # Captain: Rajat Patidar | Key: Virat Kohli, Phil Salt
    "Royal Challengers Bengaluru": {
        "batsmen": ["Virat Kohli", "Rajat Patidar", "Phil Salt", "Devdutt Padikkal", "Manoj Bhandage"],
        "bowlers": ["Josh Hazlewood", "Bhuvneshwar Kumar", "Yash Dayal", "Suyash Sharma", "Swapnil Singh"],
        "allrounders": ["Tim David", "Krunal Pandya"],
    },
    # Captain: Ajinkya Rahane | Key: Cameron Green, Venkatesh Iyer
    "Kolkata Knight Riders": {
        "batsmen": ["Ajinkya Rahane", "Venkatesh Iyer", "Angkrish Raghuvanshi", "Rinku Singh", "Manish Pandey"],
        "bowlers": ["Mitchell Starc", "Varun Chakravarthy", "Harshit Rana", "Vaibhav Arora", "Anrich Nortje"],
        "allrounders": ["Andre Russell", "Cameron Green", "Ramandeep Singh"],
    },
    # Captain: Riyan Parag | Key: Yashasvi Jaiswal, Wanindu Hasaranga
    "Rajasthan Royals": {
        "batsmen": ["Yashasvi Jaiswal", "Sanju Samson", "Shimron Hetmyer", "Rovman Powell", "Kunal Rathore"],
        "bowlers": ["Jofra Archer", "Sandeep Sharma", "Wanindu Hasaranga", "Maheesh Theekshana", "Avesh Khan"],
        "allrounders": ["Riyan Parag", "Dhruv Jurel", "Nitish Kumar Reddy"],
    },
    # Captain: Axar Patel | Key: Jake Fraser-McGurk, Faf du Plessis
    "Delhi Capitals": {
        "batsmen": ["Jake Fraser-McGurk", "Faf du Plessis", "Harry Brook", "Karun Nair", "Tristan Stubbs"],
        "bowlers": ["Mitchell Starc", "Kuldeep Yadav", "Mukesh Kumar", "Mohit Sharma", "Dushmantha Chameera"],
        "allrounders": ["Axar Patel", "Ashutosh Sharma", "Vipraj Nigam"],
    },
    # Captain: Pat Cummins | Key: Travis Head, Heinrich Klaasen
    "Sunrisers Hyderabad": {
        "batsmen": ["Travis Head", "Abhishek Sharma", "Heinrich Klaasen", "Ishan Kishan", "Adam Rossington"],
        "bowlers": ["Pat Cummins", "T Natarajan", "Harshal Patel", "Simarjeet Singh", "Zeeshan Ansari"],
        "allrounders": ["Washington Sundar", "Shahbaz Ahmed", "Kamindu Mendis"],
    },
    # Captain: Shreyas Iyer | Key: Arshdeep Singh, Glenn Maxwell
    "Punjab Kings": {
        "batsmen": ["Shreyas Iyer", "Prabhsimran Singh", "Josh Inglis", "Nehal Wadhera", "Musheer Khan"],
        "bowlers": ["Arshdeep Singh", "Marco Jansen", "Yuzvendra Chahal", "Rahul Chahar", "Harpreet Brar"],
        "allrounders": ["Glenn Maxwell", "Marcus Stoinis", "Azmatullah Omarzai"],
    },
    # Captain: Shubman Gill | Key: Jos Buttler, Rashid Khan
    "Gujarat Titans": {
        "batsmen": ["Shubman Gill", "Jos Buttler", "Sai Sudharsan", "David Miller", "Sherfane Rutherford"],
        "bowlers": ["Rashid Khan", "Mohammed Siraj", "Gerald Coetzee", "Noor Ahmad", "Ishant Sharma"],
        "allrounders": ["Rahul Tewatia", "Dasun Shanaka", "Shahrukh Khan"],
    },
    # Captain: Rishabh Pant | Key: Nicholas Pooran, Ravi Bishnoi
    "Lucknow Super Giants": {
        "batsmen": ["Rishabh Pant", "Nicholas Pooran", "Ayush Badoni", "Mitchell Marsh", "Aiden Markram"],
        "bowlers": ["Mayank Yadav", "Ravi Bishnoi", "Yash Thakur", "Mohsin Khan", "David Wiese"],
        "allrounders": ["Abdul Samad", "Kyle Mayers", "Deepak Hooda"],
    },
}


# Team strength ratings (for realistic win probability generation)
# Updated team strengths for 2026 season based on squad composition
TEAM_STRENGTHS = {
    "Mumbai Indians": 0.70,
    "Chennai Super Kings": 0.68,
    "Royal Challengers Bengaluru": 0.63,
    "Kolkata Knight Riders": 0.65,
    "Rajasthan Royals": 0.62,
    "Delhi Capitals": 0.60,
    "Sunrisers Hyderabad": 0.67,
    "Punjab Kings": 0.58,
    "Gujarat Titans": 0.66,
    "Lucknow Super Giants": 0.64,
    "Deccan Chargers": 0.48,
    "Kochi Tuskers Kerala": 0.35,
    "Pune Warriors India": 0.38,
    "Rising Pune Supergiant": 0.52,
    "Gujarat Lions": 0.50,
}

# Home advantage by venue (team -> venue -> bonus)
HOME_VENUES = {
    "Mumbai Indians": ["Wankhede Stadium, Mumbai"],
    "Chennai Super Kings": ["M. A. Chidambaram Stadium, Chennai"],
    "Royal Challengers Bengaluru": ["M. Chinnaswamy Stadium, Bengaluru"],
    "Kolkata Knight Riders": ["Eden Gardens, Kolkata"],
    "Rajasthan Royals": ["Sawai Mansingh Stadium, Jaipur"],
    "Delhi Capitals": ["Arun Jaitley Stadium, Delhi"],
    "Sunrisers Hyderabad": ["Rajiv Gandhi Intl. Cricket Stadium, Hyderabad"],
    "Punjab Kings": ["Punjab Cricket Association Stadium, Mohali"],
    "Gujarat Titans": ["Narendra Modi Stadium, Ahmedabad"],
    "Lucknow Super Giants": ["Ekana Cricket Stadium, Lucknow"],
}


def _get_teams_for_season(season: int) -> list:
    """Return the list of teams that participated in a given IPL season."""
    teams = list(CURRENT_TEAMS)

    # Historical team adjustments
    if season <= 2012:
        teams = [t for t in teams if t not in ["Gujarat Titans", "Lucknow Super Giants"]]
        if season <= 2011:
            teams.append("Kochi Tuskers Kerala")
        if season <= 2013:
            teams.append("Deccan Chargers")
            teams = [t for t in teams if t != "Sunrisers Hyderabad"]
        if season >= 2011 and season <= 2013:
            teams.append("Pune Warriors India")
    elif season in [2016, 2017]:
        teams = [t for t in teams if t not in ["Chennai Super Kings", "Rajasthan Royals",
                                                  "Gujarat Titans", "Lucknow Super Giants"]]
        teams.extend(["Rising Pune Supergiant", "Gujarat Lions"])
    elif season < 2022:
        teams = [t for t in teams if t not in ["Gujarat Titans", "Lucknow Super Giants"]]

    # Ensure exactly 8-10 teams
    return teams[:10]


def _simulate_match_result(team1: str, team2: str, venue: str, toss_winner: str, toss_decision: str) -> dict:
    """Simulate a match result based on team strengths and conditions."""
    s1 = TEAM_STRENGTHS.get(team1, 0.50)
    s2 = TEAM_STRENGTHS.get(team2, 0.50)

    # Home advantage
    home_bonus = 0.05
    if team1 in HOME_VENUES and venue in HOME_VENUES[team1]:
        s1 += home_bonus
    if team2 in HOME_VENUES and venue in HOME_VENUES[team2]:
        s2 += home_bonus

    # Toss advantage (slight)
    if toss_winner == team1:
        s1 += 0.02
    else:
        s2 += 0.02

    # Calculate win probability
    p1 = s1 / (s1 + s2)
    winner = team1 if random.random() < p1 else team2
    loser = team2 if winner == team1 else team1

    # Win margin
    if random.random() < 0.55:  # Bat first wins
        win_by_runs = random.choice(range(1, 85))
        return {"winner": winner, "win_by_runs": win_by_runs, "win_by_wickets": 0}
    else:
        win_by_wickets = random.choice(range(1, 10))
        return {"winner": winner, "win_by_runs": 0, "win_by_wickets": win_by_wickets}


def generate_matches_data(num_seasons: int = 19) -> pd.DataFrame:
    """
    Generate synthetic IPL match data from 2008 to 2008+num_seasons-1.

    Returns:
        pd.DataFrame with columns matching standard IPL dataset format.
    """
    records = []
    match_id = 1

    for season_offset in range(num_seasons):
        season = SEASON_START_YEAR + season_offset
        teams = _get_teams_for_season(season)
        num_matches = random.randint(56, 74)  # Matches per season

        season_start = datetime(season, 3, 20) + timedelta(days=random.randint(0, 10))

        for match_num in range(num_matches):
            team1, team2 = random.sample(teams, 2)
            venue = random.choice(VENUES)
            match_date = season_start + timedelta(days=match_num * random.randint(1, 3))
            toss_winner = random.choice([team1, team2])
            toss_decision = random.choice(["bat", "field"])

            result = _simulate_match_result(team1, team2, venue, toss_winner, toss_decision)

            # Player of match
            winner_team = result["winner"]
            winner_key = winner_team if winner_team in PLAYER_ARCHETYPES else random.choice(CURRENT_TEAMS)
            all_players = (
                PLAYER_ARCHETYPES.get(winner_key, {}).get("batsmen", ["Unknown"]) +
                PLAYER_ARCHETYPES.get(winner_key, {}).get("allrounders", [])
            )
            player_of_match = random.choice(all_players) if all_players else "Unknown"

            records.append({
                "match_id": match_id,
                "season": season,
                "date": match_date.strftime("%Y-%m-%d"),
                "team1": team1,
                "team2": team2,
                "venue": venue,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "winner": result["winner"],
                "win_by_runs": result["win_by_runs"],
                "win_by_wickets": result["win_by_wickets"],
                "player_of_match": player_of_match,
                "city": venue.split(",")[-1].strip() if "," in venue else "Unknown",
            })
            match_id += 1

    return pd.DataFrame(records)


def generate_deliveries_data(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic ball-by-ball delivery data for all matches.

    Returns:
        pd.DataFrame with delivery-level data.
    """
    all_deliveries = []

    for _, match in matches_df.iterrows():
        mid = match["match_id"]
        team1, team2 = match["team1"], match["team2"]

        for inning in [1, 2]:
            batting_team = team1 if inning == 1 else team2
            bowling_team = team2 if inning == 1 else team1

            # Get players
            bat_key = batting_team if batting_team in PLAYER_ARCHETYPES else random.choice(CURRENT_TEAMS)
            bowl_key = bowling_team if bowling_team in PLAYER_ARCHETYPES else random.choice(CURRENT_TEAMS)

            batsmen = PLAYER_ARCHETYPES.get(bat_key, {}).get("batsmen", ["Batsman1", "Batsman2"]) + \
                      PLAYER_ARCHETYPES.get(bat_key, {}).get("allrounders", [])
            bowlers = PLAYER_ARCHETYPES.get(bowl_key, {}).get("bowlers", ["Bowler1", "Bowler2"])

            wickets_fallen = 0
            total_runs = 0
            target_total = random.randint(130, 220) if inning == 1 else None

            for over in range(20):
                bowler = random.choice(bowlers)
                for ball in range(1, 7):
                    if wickets_fallen >= 10:
                        break

                    batsman = batsmen[min(wickets_fallen, len(batsmen) - 1)]
                    non_striker = batsmen[min(wickets_fallen + 1, len(batsmen) - 1)]

                    # Generate ball outcome
                    rand = random.random()
                    if rand < 0.35:
                        batsman_runs = 0
                    elif rand < 0.55:
                        batsman_runs = 1
                    elif rand < 0.65:
                        batsman_runs = 2
                    elif rand < 0.70:
                        batsman_runs = 3
                    elif rand < 0.82:
                        batsman_runs = 4
                    elif rand < 0.88:
                        batsman_runs = 6
                    else:
                        batsman_runs = 0

                    # Extras
                    extra_runs = 0
                    extras_type = ""
                    if random.random() < 0.05:
                        extra_runs = random.choice([1, 1, 1, 2, 4, 5])
                        extras_type = random.choice(["wides", "noballs", "byes", "legbyes"])

                    # Wicket
                    is_wicket = 0
                    dismissal_kind = ""
                    if rand >= 0.88 and wickets_fallen < 10:
                        is_wicket = 1
                        wickets_fallen += 1
                        dismissal_kind = random.choice([
                            "caught", "caught", "caught", "bowled", "bowled",
                            "lbw", "run out", "stumped", "caught and bowled"
                        ])

                    total_runs += batsman_runs + extra_runs

                    all_deliveries.append({
                        "match_id": mid,
                        "inning": inning,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "over": over,
                        "ball": ball,
                        "batsman": batsman,
                        "non_striker": non_striker,
                        "bowler": bowler,
                        "batsman_runs": batsman_runs,
                        "extra_runs": extra_runs,
                        "total_runs": batsman_runs + extra_runs,
                        "extras_type": extras_type,
                        "is_wicket": is_wicket,
                        "dismissal_kind": dismissal_kind,
                    })

                if wickets_fallen >= 10:
                    break

    return pd.DataFrame(all_deliveries)


def generate_players_data() -> pd.DataFrame:
    """Generate a players reference table."""
    records = []
    pid = 1
    for team, roles in PLAYER_ARCHETYPES.items():
        seen = set()
        for role, players in roles.items():
            for name in players:
                if name not in seen:
                    seen.add(name)
                    records.append({
                        "player_id": pid,
                        "player_name": name,
                        "team": team,
                        "role": role.rstrip("s"),  # batsmen -> batsman
                        "batting_avg": round(np.random.normal(28, 8), 1),
                        "bowling_avg": round(np.random.normal(28, 10), 1) if role != "batsmen" else None,
                        "strike_rate": round(np.random.normal(135, 20), 1),
                        "economy_rate": round(np.random.normal(8.0, 1.5), 2) if role != "batsmen" else None,
                    })
                    pid += 1
    return pd.DataFrame(records)


def generate_all_data():
    """Generate all datasets and save to data/ directory."""
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generating IPL match data...")
    matches = generate_matches_data()
    matches.to_csv(os.path.join(DATA_DIR, "matches.csv"), index=False)
    print(f"   OK: {len(matches)} matches generated -> data/matches.csv")

    print("Generating ball-by-ball delivery data...")
    deliveries = generate_deliveries_data(matches)
    deliveries.to_csv(os.path.join(DATA_DIR, "deliveries.csv"), index=False)
    print(f"   OK: {len(deliveries)} deliveries generated -> data/deliveries.csv")

    print("Generating player profiles...")
    players = generate_players_data()
    players.to_csv(os.path.join(DATA_DIR, "players.csv"), index=False)
    print(f"   OK: {len(players)} players generated -> data/players.csv")

    print("\nAll data generated successfully!")
    return matches, deliveries, players


if __name__ == "__main__":
    generate_all_data()
