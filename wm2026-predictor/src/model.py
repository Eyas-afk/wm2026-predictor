"""
Trainiert ein Modell, das aus ELO-Differenz + Kontext-Features die
Wahrscheinlichkeit für Heimsieg / Unentschieden / Auswärtssieg vorhersagt.

Modell: Gradient Boosting Classifier (XGBoost) mit 3 Klassen.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Get project root (two levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent

RESULT_MAP = {"H": 0, "D": 1, "A": 2}  # Heimsieg, Draw, Auswärtssieg


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()

    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)

    # Zielvariable
    def result(row):
        if row["home_score"] > row["away_score"]:
            return "H"
        elif row["home_score"] == row["away_score"]:
            return "D"
        else:
            return "A"

    df["result"] = df.apply(result, axis=1)
    df["target"] = df["result"].map(RESULT_MAP)

    return df


def main():
    history = pd.read_csv(PROJECT_ROOT / "data" / "elo_history.csv")
    df = build_features(history)

    features = ["elo_diff", "home_elo_pre", "away_elo_pre", "neutral"]
    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    print("=== Modell-Evaluation (Testset) ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"Log Loss: {log_loss(y_test, probs):.3f}")
    print()
    print(classification_report(y_test, preds, target_names=["Heimsieg", "Unentschieden", "Auswärtssieg"]))

    # Baseline-Vergleich: einfach immer "Heimsieg" tippen
    baseline_preds = np.zeros_like(y_test)
    print(f"Baseline (immer Heimsieg) Accuracy: {accuracy_score(y_test, baseline_preds):.3f}")

    # Feature Importance
    print("\nFeature Importance:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:15s} {imp:.3f}")

    joblib.dump(model, PROJECT_ROOT / "data" / "match_model.pkl")
    print("\n✅ Modell gespeichert: data/match_model.pkl")


if __name__ == "__main__":
    main()
