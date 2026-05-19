"""
IPL Match Prediction System - Model Training & Evaluation
==========================================================
Trains multiple classification models, evaluates them, and selects the best
performer. Includes cross-validation, hyperparameter tuning, and model persistence.

Models trained:
  1. XGBoost Classifier (primary)
  2. Random Forest Classifier
  3. Logistic Regression (baseline)
  4. Voting Ensemble (combination of all three)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

from xgboost import XGBClassifier

from config import (
    MODEL_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS,
    XGBOOST_PARAMS, RF_PARAMS, LOGISTIC_PARAMS
)
from feature_engineering import FEATURE_COLUMNS

warnings.filterwarnings("ignore")


class IPLModelTrainer:
    """
    Trains and evaluates multiple ML models for IPL match prediction.

    Usage:
        trainer = IPLModelTrainer(features_df)
        trainer.train_all()
        trainer.evaluate_all()
        trainer.save_best_model()
    """

    def __init__(self, features_df: pd.DataFrame):
        """
        Initialize trainer with feature DataFrame.

        Args:
            features_df: DataFrame from feature_engineering.build_features()
        """
        self.features_df = features_df
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None

        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        """Split features into train/test and scale."""
        X = self.features_df[FEATURE_COLUMNS].values
        y = self.features_df["target"].values

        # Handle any NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"\n📊 Data Split:")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Testing samples:  {len(self.X_test)}")
        print(f"   Features:         {len(FEATURE_COLUMNS)}")
        print(f"   Class balance:    {y.mean():.2%} Team1 wins\n")

    def _train_model(self, name: str, model, use_scaled: bool = False):
        """Train a single model and store it."""
        X_tr = self.X_train_scaled if use_scaled else self.X_train
        X_te = self.X_test_scaled if use_scaled else self.X_test

        print(f"   🏋️ Training {name}...")
        model.fit(X_tr, self.y_train)
        self.models[name] = {"model": model, "use_scaled": use_scaled}

        # Cross-validation
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_tr, self.y_train, cv=cv, scoring="accuracy")

        # Predictions
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, zero_division=0),
            "recall": recall_score(self.y_test, y_pred, zero_division=0),
            "f1": f1_score(self.y_test, y_pred, zero_division=0),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "roc_auc": roc_auc_score(self.y_test, y_prob) if y_prob is not None else None,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
        self.results[name] = metrics

        print(f"      Accuracy:  {metrics['accuracy']:.4f}")
        print(f"      F1 Score:  {metrics['f1']:.4f}")
        print(f"      CV Mean:   {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        if metrics["roc_auc"]:
            print(f"      ROC AUC:   {metrics['roc_auc']:.4f}")

    def train_all(self):
        """Train all models."""
        print("=" * 60)
        print("🚀 MODEL TRAINING")
        print("=" * 60)

        # 1. XGBoost
        xgb = XGBClassifier(**XGBOOST_PARAMS)
        self._train_model("XGBoost", xgb, use_scaled=False)
        print()

        # 2. Random Forest
        rf = RandomForestClassifier(**RF_PARAMS)
        self._train_model("Random Forest", rf, use_scaled=False)
        print()

        # 3. Logistic Regression
        lr = LogisticRegression(**LOGISTIC_PARAMS)
        self._train_model("Logistic Regression", lr, use_scaled=True)
        print()

        # 4. Voting Ensemble
        ensemble = VotingClassifier(
            estimators=[
                ("xgb", XGBClassifier(**XGBOOST_PARAMS)),
                ("rf", RandomForestClassifier(**RF_PARAMS)),
                ("lr", LogisticRegression(**LOGISTIC_PARAMS)),
            ],
            voting="soft",
            weights=[3, 2, 1],  # Weight XGBoost highest
        )
        # Ensemble uses unscaled data (XGBoost/RF don't need scaling, LR is robust enough here)
        self._train_model("Voting Ensemble", ensemble, use_scaled=False)

        # Select best model
        best_name = max(self.results, key=lambda k: self.results[k]["f1"])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]["model"]
        print(f"\n🏆 Best Model: {best_name} (F1: {self.results[best_name]['f1']:.4f})")

    def evaluate_all(self):
        """Print detailed evaluation for all models."""
        print("\n" + "=" * 60)
        print("📈 DETAILED EVALUATION RESULTS")
        print("=" * 60)

        # Summary table
        summary = []
        for name, metrics in self.results.items():
            summary.append({
                "Model": name,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1 Score": f"{metrics['f1']:.4f}",
                "CV Mean±Std": f"{metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}",
                "ROC AUC": f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A",
            })

        summary_df = pd.DataFrame(summary)
        print("\n" + summary_df.to_string(index=False))

        # Detailed classification reports
        for name, metrics in self.results.items():
            print(f"\n{'─' * 40}")
            print(f"📋 Classification Report: {name}")
            print("─" * 40)
            print(classification_report(
                self.y_test, metrics["y_pred"],
                target_names=["Team2 Wins", "Team1 Wins"]
            ))

    def plot_results(self, save_dir: str = None):
        """Generate evaluation plots."""
        save_dir = save_dir or MODEL_DIR
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("IPL Match Prediction - Model Evaluation", fontsize=16, fontweight="bold")

        # 1. Accuracy Comparison
        ax = axes[0, 0]
        names = list(self.results.keys())
        accs = [self.results[n]["accuracy"] for n in names]
        colors = sns.color_palette("viridis", len(names))
        bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        ax.set_ylim(0.4, max(accs) + 0.05)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
        ax.tick_params(axis="x", rotation=15)

        # 2. Confusion Matrix (best model)
        ax = axes[0, 1]
        cm = confusion_matrix(self.y_test, self.results[self.best_model_name]["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Team2", "Team1"], yticklabels=["Team2", "Team1"])
        ax.set_title(f"Confusion Matrix ({self.best_model_name})")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

        # 3. ROC Curves
        ax = axes[1, 0]
        for name, metrics in self.results.items():
            if metrics["y_prob"] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, metrics["y_prob"])
                auc_val = metrics["roc_auc"]
                ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax.set_title("ROC Curves")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # 4. Feature Importance (XGBoost)
        ax = axes[1, 1]
        if "XGBoost" in self.models:
            xgb_model = self.models["XGBoost"]["model"]
            importances = xgb_model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10
            top_features = [FEATURE_COLUMNS[i] for i in indices]
            top_importances = importances[indices]

            ax.barh(top_features, top_importances, color=sns.color_palette("rocket", 10))
            ax.set_title("Top 10 Feature Importances (XGBoost)")
            ax.set_xlabel("Importance")

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "model_evaluation.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n📊 Evaluation plots saved → {plot_path}")

    def save_best_model(self):
        """Save the best model and scaler to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)

        model_path = os.path.join(MODEL_DIR, "best_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        meta_path = os.path.join(MODEL_DIR, "model_meta.pkl")

        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump({
            "model_name": self.best_model_name,
            "use_scaled": self.models[self.best_model_name]["use_scaled"],
            "feature_columns": FEATURE_COLUMNS,
            "metrics": {k: v for k, v in self.results[self.best_model_name].items()
                        if k not in ["y_pred", "y_prob"]},
        }, meta_path)

        print(f"\n💾 Model saved:")
        print(f"   Model:   {model_path}")
        print(f"   Scaler:  {scaler_path}")
        print(f"   Meta:    {meta_path}")

    def save_all_models(self):
        """Save all trained models."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        for name, data in self.models.items():
            safe_name = name.lower().replace(" ", "_")
            path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
            joblib.dump(data["model"], path)
        print(f"   ✅ All models saved to {MODEL_DIR}/")
