"""
IPL Match Prediction System - Main Entry Point
================================================
Complete pipeline: data generation → feature engineering → model training
→ evaluation → prediction demo.

Usage:
    python main.py              # Full pipeline
    python main.py --predict    # Prediction mode only (requires trained model)
    python main.py --interactive # Interactive prediction mode
"""

import os
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import argparse
import time

from config import DATA_DIR, MODEL_DIR, MATCHES_CSV, DELIVERIES_CSV, CURRENT_TEAMS, VENUES


def run_full_pipeline():
    """Execute the complete training pipeline."""
    start = time.time()

    print("=" * 65)
    print("  🏏  IPL MATCH PREDICTION SYSTEM  🏏")
    print("  Complete Machine Learning Pipeline")
    print("=" * 65)

    # ──────────────────────────────────────────
    # Step 1: Data Generation / Loading
    # ──────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  STEP 1: DATA PREPARATION")
    print("─" * 65)

    if not os.path.exists(MATCHES_CSV) or not os.path.exists(DELIVERIES_CSV):
        print("📂 No existing data found. Generating synthetic IPL data...")
        from data_generator import generate_all_data
        matches_raw, deliveries_raw, players = generate_all_data()
    else:
        print("📂 Using existing data files in data/ directory.")

    # ──────────────────────────────────────────
    # Step 2: Data Loading & Exploration
    # ──────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  STEP 2: DATA EXPLORATION")
    print("─" * 65)

    from feature_engineering import load_and_clean_data
    matches, deliveries = load_and_clean_data(MATCHES_CSV, DELIVERIES_CSV)

    print(f"\n📊 Dataset Overview:")
    print(f"   Total Matches:    {len(matches)}")
    print(f"   Total Deliveries: {len(deliveries)}")
    print(f"   Seasons:          {matches['season'].min()} - {matches['season'].max()}")
    print(f"   Teams:            {matches['team1'].nunique()} unique teams")
    print(f"   Venues:           {matches['venue'].nunique()} unique venues")

    print(f"\n🏆 Top 5 Most Successful Teams:")
    team_wins = {}
    for team in set(matches["team1"].tolist() + matches["team2"].tolist()):
        team_matches = matches[(matches["team1"] == team) | (matches["team2"] == team)]
        wins = len(team_matches[team_matches["winner"] == team])
        total = len(team_matches)
        team_wins[team] = {"wins": wins, "total": total, "pct": wins / total if total > 0 else 0}

    sorted_teams = sorted(team_wins.items(), key=lambda x: x[1]["wins"], reverse=True)
    for i, (team, stats) in enumerate(sorted_teams[:5], 1):
        print(f"   {i}. {team:<35} {stats['wins']:>3}W / {stats['total']:>3}M ({stats['pct']:.1%})")

    # ──────────────────────────────────────────
    # Step 3: Feature Engineering
    # ──────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  STEP 3: FEATURE ENGINEERING")
    print("─" * 65)

    from feature_engineering import build_features
    features_df = build_features(matches, deliveries)

    print(f"\n   Feature matrix shape: {features_df.shape}")
    print(f"   Features used: {len([c for c in features_df.columns if c not in ['match_id', 'season', 'team1', 'team2', 'venue', 'target']])}")

    # Save features for inspection
    features_path = os.path.join(DATA_DIR, "features.csv")
    features_df.to_csv(features_path, index=False)
    print(f"   Features saved → {features_path}")

    # ──────────────────────────────────────────
    # Step 4: Model Training
    # ──────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  STEP 4: MODEL TRAINING & EVALUATION")
    print("─" * 65)

    from model_training import IPLModelTrainer
    trainer = IPLModelTrainer(features_df)
    trainer.train_all()
    trainer.evaluate_all()
    trainer.plot_results()
    trainer.save_best_model()
    trainer.save_all_models()

    # ──────────────────────────────────────────
    # Step 5: Prediction Demo
    # ──────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  STEP 5: PREDICTION DEMO")
    print("─" * 65)

    from predictor import IPLPredictor
    predictor = IPLPredictor()

    # Demo predictions
    demo_matches = [
        {
            "team1": "Mumbai Indians",
            "team2": "Chennai Super Kings",
            "venue": "Wankhede Stadium, Mumbai",
            "toss_winner": "Mumbai Indians",
            "toss_decision": "bat",
        },
        {
            "team1": "Royal Challengers Bengaluru",
            "team2": "Kolkata Knight Riders",
            "venue": "M. Chinnaswamy Stadium, Bengaluru",
            "toss_winner": "Kolkata Knight Riders",
            "toss_decision": "field",
        },
        {
            "team1": "Rajasthan Royals",
            "team2": "Gujarat Titans",
            "venue": "Sawai Mansingh Stadium, Jaipur",
            "toss_winner": "Rajasthan Royals",
            "toss_decision": "bat",
        },
    ]

    for match in demo_matches:
        result = predictor.predict(**match)
        predictor.display_prediction(result)

    elapsed = time.time() - start
    print(f"\n⏱️  Total pipeline time: {elapsed:.1f} seconds")
    print("✨ Pipeline complete! Model is ready for predictions.\n")


def run_prediction_mode():
    """Run in prediction-only mode with example predictions."""
    from predictor import IPLPredictor
    predictor = IPLPredictor()

    print("\n🏏 Running sample predictions...\n")

    result = predictor.predict(
        team1="Mumbai Indians",
        team2="Delhi Capitals",
        venue="Wankhede Stadium, Mumbai",
        toss_winner="Delhi Capitals",
        toss_decision="field",
    )
    predictor.display_prediction(result)


def run_interactive_mode():
    """Interactive mode for making custom predictions."""
    from predictor import IPLPredictor

    try:
        predictor = IPLPredictor()
    except FileNotFoundError:
        print("❌ No trained model found. Please run the full pipeline first:")
        print("   python main.py")
        return

    print("\n" + "=" * 65)
    print("  🏏  IPL PREDICTION - INTERACTIVE MODE  🏏")
    print("=" * 65)
    print("  Type 'quit' to exit, 'teams' to list teams, 'venues' to list venues.\n")

    while True:
        try:
            print("\n" + "─" * 50)
            team1 = input("  Enter Team 1: ").strip()
            if team1.lower() == "quit":
                break
            if team1.lower() == "teams":
                predictor.list_teams()
                continue
            if team1.lower() == "venues":
                predictor.list_venues()
                continue

            team2 = input("  Enter Team 2: ").strip()
            if team2.lower() == "quit":
                break

            print("\n  Available venues (enter number or full name):")
            predictor.list_venues()
            venue_input = input("\n  Enter venue (number or name): ").strip()

            # Handle venue by number
            try:
                venue_idx = int(venue_input) - 1
                if 0 <= venue_idx < len(VENUES):
                    venue = VENUES[venue_idx]
                else:
                    venue = venue_input
            except ValueError:
                venue = venue_input

            toss_winner = input(f"  Toss winner ({team1} / {team2}): ").strip()
            toss_decision = input("  Toss decision (bat / field): ").strip()

            result = predictor.predict(
                team1=team1, team2=team2, venue=venue,
                toss_winner=toss_winner, toss_decision=toss_decision
            )
            predictor.display_prediction(result)

        except ValueError as e:
            print(f"\n  ⚠️  Input Error: {e}")
        except KeyboardInterrupt:
            print("\n\n  👋 Goodbye!")
            break

    print("\n  👋 Thanks for using IPL Predictor!")


def main():
    parser = argparse.ArgumentParser(
        description="🏏 IPL Match Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  # Full pipeline: generate data, train, evaluate, predict
  python main.py --predict        # Run predictions with trained model
  python main.py --interactive    # Interactive prediction mode
        """
    )
    parser.add_argument("--predict", action="store_true",
                        help="Run in prediction-only mode (requires trained model)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive prediction mode")

    args = parser.parse_args()

    if args.interactive:
        run_interactive_mode()
    elif args.predict:
        run_prediction_mode()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
