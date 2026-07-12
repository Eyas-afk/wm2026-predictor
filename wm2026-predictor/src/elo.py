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

# Gewichtung je nach Turnier-Wichtigkeit (K-Faktor-Multiplikator).
# Höherer Wert = das Ergebnis verändert das Rating stärker, weil wichtige
# Turniere aussagekräftiger für die "echte" Team-Stärke sind als z.B.
# Freundschaftsspiele, in denen oft nicht die beste Aufstellung spielt.
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
# Fallback-Gewichtung für alle Turniere, die nicht explizit oben gelistet
# sind (z.B. CECAFA Cup, Gulf Cup) - Datensatz enthält 100+ Turnierarten,
# nicht praktikabel, jede einzeln zu bewerten.
DEFAULT_WEIGHT = 30

# Startwert für Teams, die zum ersten Mal in der Historie auftauchen.
# 1500 ist der übliche ELO-Standard-Startwert (neutral, weder stark noch schwach).
BASE_RATING = 1500


def get_tournament_weight(tournament: str) -> float:
    """Liefert den K-Faktor für ein Turnier, mit Fallback auf DEFAULT_WEIGHT
    falls das Turnier nicht in TOURNAMENT_WEIGHTS explizit gelistet ist."""
    return TOURNAMENT_WEIGHTS.get(tournament, DEFAULT_WEIGHT)


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Erwartete Punktzahl (0-1) von Team A gegen Team B nach der Standard-ELO-Formel.

    Beispiel: rating_a=2200, rating_b=1500 -> ~0.98 (A wird fast sicher gewinnen).
    Die Formel ist symmetrisch: expected_score(a, b) + expected_score(b, a) == 1.
    """
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def goal_diff_multiplier(goal_diff: int) -> float:
    """
    Verstärkt die Rating-Änderung bei deutlichen Siegen (z.B. 5:0 zählt mehr
    als 1:0), aber mit abnehmendem Grenzertrag - der Unterschied zwischen
    einem 7:0 und einem 8:0 sagt kaum noch etwas zusätzliches über die
    Team-Stärke aus. Standard-Ansatz aus dem World Football Elo Rating System.

    goal_diff <= 1  -> kein Bonus (normaler Sieg/Remis)
    goal_diff == 2  -> 1.5x Verstärkung
    goal_diff >= 3  -> linear wachsend, aber abflachend (11+gd)/8
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

    Wichtig: df muss VORHER chronologisch aufsteigend sortiert werden (siehe
    df_sorted unten) - die Reihenfolge ist zwingend, weil jede Rating-Änderung
    auf dem Rating unmittelbar vor diesem Spiel basiert. Falsche Reihenfolge
    würde das gesamte Rating-System verfälschen.

    Returns:
        final_ratings: dict {team_name: elo_rating} nach dem letzten Spiel
                        in der Historie - repräsentiert die aktuelle Team-Stärke.
        history: DataFrame mit den ELO-Werten beider Teams VOR jedem Spiel.
                 Das ist absichtlich getrennt von "danach"-Werten, um Data
                 Leakage beim späteren Modelltraining zu vermeiden: das Modell
                 darf nur wissen, was VOR dem Spiel bekannt war, nicht das
                 Ergebnis, das es ja gerade erst vorhersagen soll.
    """
    # Aktuelles Rating pro Team, wird bei jedem Spiel live aktualisiert.
    ratings = {}
    # Sammelt für jedes Spiel einen "Snapshot" der Ratings VOR dem Spiel -
    # das werden später die Trainings-Features für model.py.
    records = []

    # Chronologisch sortieren ist Pflicht, siehe Docstring oben.
    df_sorted = df.sort_values("date").reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        home, away = row["home_team"], row["away_team"]
        home_score, away_score = row["home_score"], row["away_score"]

        # .get() mit Fallback auf BASE_RATING, falls das Team zum ersten Mal
        # in der Historie auftaucht (sonst KeyError).
        r_home = ratings.get(home, BASE_RATING)
        r_away = ratings.get(away, BASE_RATING)

        # ELO-Werte VOR dem Spiel speichern -> das sind unsere Modell-Features.
        # Muss vor der Rating-Aktualisierung passieren!
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

        # Tatsächliches Ergebnis in ELO-Punktzahl übersetzen:
        # 1 = Heimsieg, 0.5 = Unentschieden, 0 = Auswärtssieg.
        if home_score > away_score:
            actual_home = 1.0
        elif home_score == away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        # Erwarteten Ausgang berechnen, BEVOR wir wissen was passiert ist -
        # exp_home ist quasi "was das Rating vorher schon geglaubt hat".
        exp_home = expected_score(r_home, r_away)
        goal_diff = abs(int(home_score) - int(away_score))
        weight = get_tournament_weight(row["tournament"])
        gd_mult = goal_diff_multiplier(goal_diff)

        # Kernformel: Rating-Änderung = K-Faktor * Tordifferenz-Bonus *
        # (tatsächliches Ergebnis - erwartetes Ergebnis).
        # Je größer die Überraschung (actual - expected), desto größer der Sprung.
        change = weight * gd_mult * (actual_home - exp_home)

        # ELO ist ein Nullsummensystem: was das eine Team gewinnt,
        # verliert das andere exakt in gleicher Höhe.
        ratings[home] = r_home + change
        ratings[away] = r_away - change

    history = pd.DataFrame(records)
    return ratings, history


if __name__ == "__main__":
    # Rohdaten laden und Datumsspalte in ein echtes datetime-Objekt umwandeln
    # (Text-Sortierung würde bei Datumsangaben falsche Reihenfolgen ergeben).
    df = pd.read_csv("C:\\Users\\ghbar\\Downloads\\wm2026-predictor\\wm2026-predictor\\data\\results.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Cutoff auf den Tag vor WM-2026-Start setzen: wir trainieren NUR mit
    # Daten, die zu diesem Zeitpunkt bekannt gewesen wären. So lässt sich das
    # Modell später fair gegen den tatsächlichen (uns schon bekannten)
    # Turnierverlauf validieren, ohne Data Leakage.
    cutoff = pd.Timestamp("2026-06-11")
    train_df = df[(df["date"] < cutoff) & df["home_score"].notna()].copy()

    print(f"Berechne ELO-Ratings aus {len(train_df)} Spielen...")
    final_ratings, history = compute_elo_ratings(train_df)

    # Kurzer Sanity-Check in der Konsole: sind die Top-Teams plausibel?
    top20 = sorted(final_ratings.items(), key=lambda x: -x[1])[:20]
    print("\nTop 20 Teams nach ELO-Rating (Stand: 10. Juni 2026):")
    for i, (team, rating) in enumerate(top20, 1):
        print(f"{i:2d}. {team:20s} {rating:.1f}")

    # Beide Ergebnisse persistieren, damit model.py sie weiterverwenden kann,
    # ohne die komplette ELO-Berechnung jedes Mal neu laufen zu lassen.
    history.to_csv("C:\\Users\\ghbar\\Downloads\\wm2026-predictor\\wm2026-predictor\\data\\elo_history.csv", index=False)
    pd.Series(final_ratings, name="elo").to_csv("C:\\Users\\ghbar\\Downloads\\wm2026-predictor\\wm2026-predictor\\data\\final_elo_ratings.csv")
    print("\n✅ Gespeichert: data/elo_history.csv, data/final_elo_ratings.csv")