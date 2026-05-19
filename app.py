"""
IPL Match Prediction System - Web Application
================================================
Flask web server providing a beautiful UI for IPL match predictions.

Usage:
    python app.py
    Then open http://localhost:5000 in your browser.
"""

import os
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import json
import traceback
from flask import Flask, render_template, request, jsonify
from config import CURRENT_TEAMS, VENUES, PLAYERS_CSV
import pandas as pd

# Load players and build PLAYER_ARCHETYPES (grouped by role with AI-predicted XI)
PLAYER_ARCHETYPES = {}
PLAYER_LOAD_ERROR = None
try:
    if os.path.exists(PLAYERS_CSV):
        players_df = pd.read_csv(PLAYERS_CSV)
        # Parse numeric stats properly
        players_df['batting_avg'] = pd.to_numeric(players_df['batting_avg'], errors='coerce').fillna(0.0)
        players_df['bowling_avg'] = pd.to_numeric(players_df['bowling_avg'], errors='coerce').fillna(999.0)
        players_df['strike_rate'] = pd.to_numeric(players_df['strike_rate'], errors='coerce').fillna(0.0)
        players_df['economy_rate'] = pd.to_numeric(players_df['economy_rate'], errors='coerce').fillna(99.0)

        # Calculate a recommendation score for each player
        def calculate_score(row):
            role = str(row['role']).lower()
            b_avg = row['batting_avg']
            sr = row['strike_rate']
            bowl_avg = row['bowling_avg']
            econ = row['economy_rate']

            if 'bat' in role:
                return b_avg * 1.5 + (sr / 10.0)
            elif 'bowl' in role:
                b_score = (100.0 / bowl_avg) if bowl_avg > 0 else 0.0
                e_score = (20.0 / econ) if econ > 0 else 0.0
                return b_score * 2.0 + e_score * 1.5
            else: # allrounder
                bat_part = b_avg * 0.8 + (sr / 20.0)
                bowl_part = (50.0 / bowl_avg) if bowl_avg > 0 else 0.0
                econ_part = (10.0 / econ) if econ > 0 else 0.0
                return bat_part + bowl_part + econ_part

        players_df['rec_score'] = players_df.apply(calculate_score, axis=1)

        # Select top 11 players for each team to mark as recommended Playing XI
        recommended_players = set()
        for team in players_df['team'].unique():
            team_players = players_df[players_df['team'] == team]
            top_11 = team_players.nlargest(11, 'rec_score')
            recommended_players.update(top_11['player_name'].tolist())

        for _, row in players_df.iterrows():
            team = row['team']
            role = row.get('role', 'batsmen')
            player_name = row['player_name']
            is_rec = player_name in recommended_players

            player_data = {
                "name": player_name,
                "batting_avg": float(row['batting_avg']),
                "strike_rate": float(row['strike_rate']),
                "bowling_avg": float(row['bowling_avg']) if row['bowling_avg'] != 999.0 else None,
                "economy_rate": float(row['economy_rate']) if row['economy_rate'] != 99.0 else None,
                "recommended": is_rec
            }

            if team not in PLAYER_ARCHETYPES:
                PLAYER_ARCHETYPES[team] = {'batsmen': [], 'bowlers': [], 'allrounders': []}
            
            if 'bat' in role.lower():
                PLAYER_ARCHETYPES[team]['batsmen'].append(player_data)
            elif 'bowl' in role.lower():
                PLAYER_ARCHETYPES[team]['bowlers'].append(player_data)
            else:
                PLAYER_ARCHETYPES[team]['allrounders'].append(player_data)
    else:
        PLAYER_LOAD_ERROR = f"PLAYERS_CSV file does not exist at expected path: {PLAYERS_CSV}"
except Exception as e:
    PLAYER_LOAD_ERROR = f"Exception during startup players loading: {traceback.format_exc()}"
    print(f"Warning: Could not load players data: {e}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, "templates"), 
            static_folder=os.path.join(BASE_DIR, "static"))

@app.errorhandler(500)
@app.errorhandler(Exception)
def handle_exception(e):
    """Catch all exceptions and display them so we can debug on Vercel."""
    traceback.print_exc()
    error_msg = traceback.format_exc() if isinstance(e, Exception) else str(e)
    return f"<h1>An Error Occurred</h1><pre>{error_msg}</pre>", 500



# ── Lazy-load predictor (heavy, only init once) ──
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        from predictor import IPLPredictor
        _predictor = IPLPredictor()
    return _predictor


# ── Routes ──

@app.route("/")
def index():
    """Serve the main prediction page."""
    return render_template("index.html", teams=CURRENT_TEAMS, venues=VENUES)


@app.route("/get-squad/<team>")
def get_squad(team):
    """Return the roster for a specific team, grouped by role."""
    if PLAYER_LOAD_ERROR:
        return jsonify({"error": "Startup Load Error", "details": PLAYER_LOAD_ERROR}), 500
    squad = PLAYER_ARCHETYPES.get(team, {'batsmen': [], 'bowlers': [], 'allrounders': []})
    return jsonify({"players": squad})


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for match prediction."""
    try:
        data = request.get_json()

        team1 = data.get("team1", "").strip()
        team2 = data.get("team2", "").strip()
        venue = data.get("venue", "").strip()
        toss_winner = data.get("toss_winner", "").strip()
        toss_decision = data.get("toss_decision", "").strip()

        squad1 = data.get("squad1", [])
        squad2 = data.get("squad2", [])

        predictor = get_predictor()
        result = predictor.predict(
            team1=team1,
            team2=team2,
            venue=venue,
            toss_winner=toss_winner,
            toss_decision=toss_decision,
            include_players=True,
            squad1=squad1,
            squad2=squad2,
        )

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": f"Model not found. Run 'python main.py' first to train the model. Details: {e}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/teams")
def teams():
    """Return available teams."""
    return jsonify({"teams": CURRENT_TEAMS})


@app.route("/venues")
def venues():
    """Return available venues."""
    return jsonify({"venues": VENUES})


# ── IPL 2026 Scorecard Data ──
IPL_2026_SCORECARDS = [
    {
        "match_no": 1, "date": "22 Mar 2026",
        "team1": "Kolkata Knight Riders", "team2": "Royal Challengers Bengaluru",
        "venue": "Eden Gardens, Kolkata",
        "inn1": {"team": "Kolkata Knight Riders", "score": "174/6", "overs": "20.0",
                 "top_bat": "Angkrish Raghuvanshi 68(42)", "top_bowl": "Yash Dayal 2/28"},
        "inn2": {"team": "Royal Challengers Bengaluru", "score": "169/8", "overs": "20.0",
                 "top_bat": "Virat Kohli 72(54)", "top_bowl": "Varun Chakravarthy 3/31"},
        "result": "Kolkata Knight Riders won by 5 runs", "potm": "Angkrish Raghuvanshi",
    },
    {
        "match_no": 2, "date": "23 Mar 2026",
        "team1": "Chennai Super Kings", "team2": "Mumbai Indians",
        "venue": "M. A. Chidambaram Stadium, Chennai",
        "inn1": {"team": "Chennai Super Kings", "score": "192/4", "overs": "20.0",
                 "top_bat": "Ruturaj Gaikwad 91(58)", "top_bowl": "Jasprit Bumrah 2/34"},
        "inn2": {"team": "Mumbai Indians", "score": "185/7", "overs": "20.0",
                 "top_bat": "Rohit Sharma 76(48)", "top_bowl": "Matheesha Pathirana 3/29"},
        "result": "Chennai Super Kings won by 7 runs", "potm": "Ruturaj Gaikwad",
    },
    {
        "match_no": 3, "date": "24 Mar 2026",
        "team1": "Rajasthan Royals", "team2": "Punjab Kings",
        "venue": "Sawai Mansingh Stadium, Jaipur",
        "inn1": {"team": "Punjab Kings", "score": "187/5", "overs": "20.0",
                 "top_bat": "Shreyas Iyer 83(52)", "top_bowl": "Jofra Archer 2/31"},
        "inn2": {"team": "Rajasthan Royals", "score": "188/3", "overs": "18.4",
                 "top_bat": "Yashasvi Jaiswal 95(60)", "top_bowl": "Arshdeep Singh 2/35"},
        "result": "Rajasthan Royals won by 7 wickets", "potm": "Yashasvi Jaiswal",
    },
    {
        "match_no": 4, "date": "25 Mar 2026",
        "team1": "Sunrisers Hyderabad", "team2": "Gujarat Titans",
        "venue": "Rajiv Gandhi Intl. Cricket Stadium, Hyderabad",
        "inn1": {"team": "Sunrisers Hyderabad", "score": "218/4", "overs": "20.0",
                 "top_bat": "Travis Head 112(56)", "top_bowl": "Rashid Khan 2/38"},
        "inn2": {"team": "Gujarat Titans", "score": "201/7", "overs": "20.0",
                 "top_bat": "Jos Buttler 78(44)", "top_bowl": "Pat Cummins 3/41"},
        "result": "Sunrisers Hyderabad won by 17 runs", "potm": "Travis Head",
    },
    {
        "match_no": 5, "date": "26 Mar 2026",
        "team1": "Delhi Capitals", "team2": "Lucknow Super Giants",
        "venue": "Arun Jaitley Stadium, Delhi",
        "inn1": {"team": "Delhi Capitals", "score": "163/8", "overs": "20.0",
                 "top_bat": "Harry Brook 58(41)", "top_bowl": "Mayank Yadav 3/26"},
        "inn2": {"team": "Lucknow Super Giants", "score": "165/4", "overs": "18.2",
                 "top_bat": "Rishabh Pant 71(44)", "top_bowl": "Kuldeep Yadav 2/29"},
        "result": "Lucknow Super Giants won by 6 wickets", "potm": "Rishabh Pant",
    },
    {
        "match_no": 6, "date": "27 Mar 2026",
        "team1": "Mumbai Indians", "team2": "Gujarat Titans",
        "venue": "Wankhede Stadium, Mumbai",
        "inn1": {"team": "Mumbai Indians", "score": "196/6", "overs": "20.0",
                 "top_bat": "Suryakumar Yadav 88(47)", "top_bowl": "Rashid Khan 2/33"},
        "inn2": {"team": "Gujarat Titans", "score": "178/9", "overs": "20.0",
                 "top_bat": "Shubman Gill 64(45)", "top_bowl": "Jasprit Bumrah 4/22"},
        "result": "Mumbai Indians won by 18 runs", "potm": "Jasprit Bumrah",
    },
    {
        "match_no": 7, "date": "28 Mar 2026",
        "team1": "Royal Challengers Bengaluru", "team2": "Rajasthan Royals",
        "venue": "M. Chinnaswamy Stadium, Bengaluru",
        "inn1": {"team": "Royal Challengers Bengaluru", "score": "210/3", "overs": "20.0",
                 "top_bat": "Virat Kohli 96(62)", "top_bowl": "Jofra Archer 2/38"},
        "inn2": {"team": "Rajasthan Royals", "score": "198/6", "overs": "20.0",
                 "top_bat": "Sanju Samson 79(48)", "top_bowl": "Josh Hazlewood 3/37"},
        "result": "Royal Challengers Bengaluru won by 12 runs", "potm": "Virat Kohli",
    },
    {
        "match_no": 8, "date": "29 Mar 2026",
        "team1": "Lucknow Super Giants", "team2": "Kolkata Knight Riders",
        "venue": "Ekana Cricket Stadium, Lucknow",
        "inn1": {"team": "Kolkata Knight Riders", "score": "181/7", "overs": "20.0",
                 "top_bat": "Andre Russell 65(32)", "top_bowl": "Ravi Bishnoi 3/29"},
        "inn2": {"team": "Lucknow Super Giants", "score": "182/5", "overs": "19.3",
                 "top_bat": "Nicholas Pooran 74(40)", "top_bowl": "Varun Chakravarthy 2/35"},
        "result": "Lucknow Super Giants won by 5 wickets", "potm": "Nicholas Pooran",
    },
]

# ── IPL 2026 Points Table ──
IPL_2026_POINTS = [
    {"pos": 1, "team": "Sunrisers Hyderabad", "p": 8, "w": 6, "l": 2, "nr": 0, "pts": 12, "nrr": "+1.082", "captain": "Pat Cummins", "form": ["W","W","L","W","W"]},
    {"pos": 2, "team": "Gujarat Titans",      "p": 8, "w": 6, "l": 2, "nr": 0, "pts": 12, "nrr": "+0.754", "captain": "Shubman Gill", "form": ["W","L","W","W","L"]},
    {"pos": 3, "team": "Lucknow Super Giants","p": 8, "w": 5, "l": 3, "nr": 0, "pts": 10, "nrr": "+0.621", "captain": "Rishabh Pant", "form": ["W","W","W","L","W"]},
    {"pos": 4, "team": "Royal Challengers Bengaluru","p": 8, "w": 5, "l": 3, "nr": 0, "pts": 10, "nrr": "+0.432", "captain": "Rajat Patidar", "form": ["L","W","W","L","W"]},
    {"pos": 5, "team": "Mumbai Indians",      "p": 8, "w": 5, "l": 3, "nr": 0, "pts": 10, "nrr": "+0.213", "captain": "Hardik Pandya", "form": ["W","L","W","W","L"]},
    {"pos": 6, "team": "Chennai Super Kings", "p": 8, "w": 4, "l": 4, "nr": 0, "pts":  8, "nrr": "+0.187", "captain": "Ruturaj Gaikwad", "form": ["W","W","L","L","W"]},
    {"pos": 7, "team": "Kolkata Knight Riders","p": 8, "w": 4, "l": 4, "nr": 0, "pts":  8, "nrr": "-0.042", "captain": "Ajinkya Rahane", "form": ["W","L","L","W","L"]},
    {"pos": 8, "team": "Rajasthan Royals",    "p": 8, "w": 3, "l": 5, "nr": 0, "pts":  6, "nrr": "-0.234", "captain": "Riyan Parag", "form": ["L","W","L","L","W"]},
    {"pos": 9, "team": "Punjab Kings",        "p": 8, "w": 3, "l": 5, "nr": 0, "pts":  6, "nrr": "-0.487", "captain": "Shreyas Iyer", "form": ["L","L","W","L","W"]},
    {"pos": 10,"team": "Delhi Capitals",      "p": 8, "w": 1, "l": 7, "nr": 0, "pts":  2, "nrr": "-2.526", "captain": "Axar Patel", "form": ["L","L","L","W","L"]},
]


@app.route("/scorecard")
def scorecard():
    """Serve the IPL 2026 scorecard page."""
    return render_template("scorecard.html", scorecards=IPL_2026_SCORECARDS)


@app.route("/points-table")
def points_table():
    """Serve the IPL 2026 points table page."""
    return render_template("points_table.html", points=IPL_2026_POINTS)


if __name__ == "__main__":
    print("\n IPL Match Prediction - Web Server")
    print("=" * 50)
    print(" Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
