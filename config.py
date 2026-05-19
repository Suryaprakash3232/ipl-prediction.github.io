"""
IPL Match Prediction System - Configuration
=============================================
Central configuration for paths, team mappings, and model hyperparameters.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MATCHES_CSV = os.path.join(DATA_DIR, "matches.csv")
DELIVERIES_CSV = os.path.join(DATA_DIR, "deliveries.csv")
PLAYERS_CSV = os.path.join(DATA_DIR, "players.csv")

# ──────────────────────────────────────────────
# IPL Teams (current + historical)
# ──────────────────────────────────────────────
CURRENT_TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Delhi Capitals",
    "Sunrisers Hyderabad",
    "Punjab Kings",
    "Gujarat Titans",
    "Lucknow Super Giants",
]

HISTORICAL_TEAMS = CURRENT_TEAMS + [
    "Deccan Chargers",
    "Kochi Tuskers Kerala",
    "Pune Warriors India",
    "Rising Pune Supergiant",
    "Gujarat Lions",
]

# Team name normalization (handles name changes over the years)
TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Deccan Chargers": "Sunrisers Hyderabad",
}

# ──────────────────────────────────────────────
# Venues
# ──────────────────────────────────────────────
VENUES = [
    "Wankhede Stadium, Mumbai",
    "M. A. Chidambaram Stadium, Chennai",
    "Eden Gardens, Kolkata",
    "M. Chinnaswamy Stadium, Bengaluru",
    "Arun Jaitley Stadium, Delhi",
    "Rajiv Gandhi Intl. Cricket Stadium, Hyderabad",
    "Sawai Mansingh Stadium, Jaipur",
    "Punjab Cricket Association Stadium, Mohali",
    "Narendra Modi Stadium, Ahmedabad",
    "Ekana Cricket Stadium, Lucknow",
    "DY Patil Stadium, Mumbai",
    "Brabourne Stadium, Mumbai",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala",
    "Holkar Cricket Stadium, Indore",
    "Maharashtra Cricket Association Stadium, Pune",
]

# ──────────────────────────────────────────────
# Model Hyperparameters
# ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LOGISTIC_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

# ──────────────────────────────────────────────
# Feature Engineering Settings
# ──────────────────────────────────────────────
RECENT_FORM_WINDOW = 5      # Number of recent matches for form calculation
H2H_MIN_MATCHES = 3         # Minimum head-to-head matches for reliable stats
SEASON_START_YEAR = 2008     # First IPL season
