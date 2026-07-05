"""
ELO-Rating-System für Nationalmannschaften.

Berechnet ein fortlaufendes ELO-Rating für jedes Team basierend auf der
gesamten historischen Spieldatenbank. Das Rating berücksichtigt:
- Sieg/Unentschieden/Niederlage
- Torunterschied (größere Siege = größerer Rating-Sprung)
- Wichtigkeit des Turniers (WM > Kontinental-Cup > Quali > Freundschaftsspiel)

Basiert auf dem bekannten "World Football Elo Ratings"-Ansatz.
"""

import pandas as pd
import numpy as np

# Gewichtung je nach Turnier-Wichtigkeit (K-Faktor-Multiplikator)
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 35,
    "Copa América": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "UEFA Nations League": 35,
    "Friendly": 20,
}
DEFAULT_WEIGHT = 30  # für alle nicht explizit gelisteten Turniere

BASE_RATING = 1500


def get_tournament_weight(tournament: str) -> float:
    return TOURNAMENT_WEIGHTS.get(tournament, DEFAULT_WEIGHT)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Erwartete Punktzahl (0-1) von Team A gegen Team B nach ELO-Formel."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def goal_diff_multiplier(goal_diff: int) -> float:
    """
    Größere Siege zählen mehr, aber mit abnehmendem Grenzertrag
    (Standard-Ansatz aus dem World Football Elo Rating System).
    """
    if goal_diff <= 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return (11 + goal_diff) / 8


def compute_elo_ratings(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Berechnet ELO-Ratings für alle Teams über die gesamte Spielhistorie.

    Returns:
        final_ratings: dict {team_name: elo_rating} nach dem letzten Spiel
        history: DataFrame mit ELO-Werten beider Teams VOR jedem Spiel
                 (wichtig fürs Modelltraining, um Data Leakage zu vermeiden!)
    """
    ratings = {}
    records = []

    df_sorted = df.sort_values("date").reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        home, away = row["home_team"], row["away_team"]
        home_score, away_score = row["home_score"], row["away_score"]

        r_home = ratings.get(home, BASE_RATING)
        r_away = ratings.get(away, BASE_RATING)

        # ELO-Werte VOR dem Spiel speichern -> das sind unsere Modell-Features
        records.append({
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            "home_elo_pre": r_home,
            "away_elo_pre": r_away,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": row["tournament"],
            "neutral": row["neutral"],
        })

        # Tatsächliches Ergebnis (1 = Heimsieg, 0.5 = Unentschieden, 0 = Auswärtssieg)
        if home_score > away_score:
            actual_home = 1.0
        elif home_score == away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        exp_home = expected_score(r_home, r_away)
        goal_diff = abs(int(home_score) - int(away_score))
        weight = get_tournament_weight(row["tournament"])
        gd_mult = goal_diff_multiplier(goal_diff)

        change = weight * gd_mult * (actual_home - exp_home)

        ratings[home] = r_home + change
        ratings[away] = r_away - change

    history = pd.DataFrame(records)
    return ratings, history


if __name__ == "__main__":
    df = pd.read_csv("data/results.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Nur Spiele vor WM-2026-Start verwenden (kein Data Leakage!)
    cutoff = pd.Timestamp("2026-06-11")
    train_df = df[(df["date"] < cutoff) & df["home_score"].notna()].copy()

    print(f"Berechne ELO-Ratings aus {len(train_df)} Spielen...")
    final_ratings, history = compute_elo_ratings(train_df)

    # Top 20 Teams nach ELO ausgeben
    top20 = sorted(final_ratings.items(), key=lambda x: -x[1])[:20]
    print("\nTop 20 Teams nach ELO-Rating (Stand: 10. Juni 2026):")
    for i, (team, rating) in enumerate(top20, 1):
        print(f"{i:2d}. {team:20s} {rating:.1f}")

    # Speichern für die nächsten Schritte
    history.to_csv("data/elo_history.csv", index=False)
    pd.Series(final_ratings, name="elo").to_csv("data/final_elo_ratings.csv")
    print("\n✅ Gespeichert: data/elo_history.csv, data/final_elo_ratings.csv")
