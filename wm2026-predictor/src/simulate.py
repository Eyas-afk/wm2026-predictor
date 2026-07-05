"""
Simuliert die WM 2026 (48 Teams, 12 Gruppen à 4) per Monte-Carlo-Verfahren,
basierend auf dem trainierten Modell und den ELO-Ratings vom 10. Juni 2026
(letzter Tag vor Turnierstart -> kein Data Leakage).

Format: 12 Gruppen -> Gruppensieger + Gruppenzweite (24) + 8 beste Dritte
        -> Achtundzwanzig... nein: 24 + 8 = 32 Teams -> K.o.-Runde (Runde
        der letzten 32 -> Achtelfinale -> Viertelfinale -> Halbfinale -> Finale)
"""

import pandas as pd
import numpy as np
import joblib
import json
from collections import defaultdict

N_SIMULATIONS = 10_000
RNG = np.random.default_rng(42)

# Die 12 Gruppen, wie im echten WM-2026-Turnier (aus den historischen Daten rekonstruiert)
GROUPS = {
    "A": ["Algeria", "Argentina", "Austria", "Jordan"],
    "B": ["Australia", "Paraguay", "Turkey", "United States"],
    "C": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "D": ["Bosnia and Herzegovina", "Canada", "Qatar", "Switzerland"],
    "E": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "F": ["Cape Verde", "Saudi Arabia", "Spain", "Uruguay"],
    "G": ["Colombia", "DR Congo", "Portugal", "Uzbekistan"],
    "H": ["Croatia", "England", "Ghana", "Panama"],
    "I": ["Curaçao", "Ecuador", "Germany", "Ivory Coast"],
    "J": ["Czech Republic", "Mexico", "South Africa", "South Korea"],
    "K": ["France", "Iraq", "Norway", "Senegal"],
    "L": ["Japan", "Netherlands", "Sweden", "Tunisia"],
}


def load_elo():
    elo = pd.read_csv("data/final_elo_ratings.csv", index_col=0).iloc[:, 0].to_dict()
    return elo


def precompute_all_probs(model, elo, teams):
    """
    Berechnet EINMAL alle paarweisen Wahrscheinlichkeiten für alle Teams im
    Turnier (statt bei jeder Simulation neu). Das ist der Performance-Trick,
    der 10.000 Simulationen erst praktikabel macht.
    """
    rows = []
    pairs = []
    for a in teams:
        for b in teams:
            if a == b:
                continue
            ra, rb = elo.get(a, 1500), elo.get(b, 1500)
            rows.append({"elo_diff": ra - rb, "home_elo_pre": ra, "away_elo_pre": rb, "neutral": 1})
            pairs.append((a, b))

    X = pd.DataFrame(rows)
    probs = model.predict_proba(X)  # (n, 3) -> [p_home, p_draw, p_away]

    prob_table = {}
    for (a, b), (p_a, p_d, p_b) in zip(pairs, probs):
        prob_table[(a, b)] = (p_a, p_d, p_b)
    return prob_table


def simulate_match_score(prob_table, team_a, team_b):
    """Simuliert Ausgang eines K.o.-Spiels (kein Unentschieden möglich -> Elfmeterschießen)."""
    p_a, p_d, p_b = prob_table[(team_a, team_b)]
    p_a_wins_knockout = p_a + p_d / 2
    return team_a if RNG.random() < p_a_wins_knockout else team_b


def simulate_group(prob_table, teams):
    """Simuliert alle 6 Spiele einer Gruppe, gibt Tabelle (Punkte) zurück."""
    points = {t: 0 for t in teams}
    goal_diff = {t: 0 for t in teams}  # vereinfachte Tordifferenz-Heuristik für Tie-Break

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            p_a, p_d, p_b = prob_table[(a, b)]
            outcome = RNG.choice(["A", "D", "B"], p=[p_a, p_d, p_b])
            if outcome == "A":
                points[a] += 3
                goal_diff[a] += 1
                goal_diff[b] -= 1
            elif outcome == "B":
                points[b] += 3
                goal_diff[b] += 1
                goal_diff[a] -= 1
            else:
                points[a] += 1
                points[b] += 1

    ranking = sorted(teams, key=lambda t: (points[t], goal_diff[t]), reverse=True)
    return ranking, points, goal_diff


def run_single_tournament(prob_table):
    group_results = {}
    all_third_placed = []

    for g_name, teams in GROUPS.items():
        ranking, points, gd = simulate_group(prob_table, teams)
        group_results[g_name] = ranking
        all_third_placed.append((ranking[2], points[ranking[2]], gd[ranking[2]]))

    # 8 beste Gruppendritte weiterkommen
    best_thirds = sorted(all_third_placed, key=lambda x: (x[1], x[2]), reverse=True)[:8]
    best_third_teams = [t[0] for t in best_thirds]

    round_of_32 = []
    for ranking in group_results.values():
        round_of_32.extend([ranking[0], ranking[1]])
    round_of_32.extend(best_third_teams)

    # Vereinfachtes K.o.-Bracket: zufällige aber feste Paarung (echtes Bracket ist
    # komplex, hier fürs Modell-Prinzip vereinfacht)
    bracket = list(round_of_32)
    RNG.shuffle(bracket)

    round_name_progress = defaultdict(list)
    current_round = bracket
    round_names = ["R32", "R16", "QF", "SF", "F"]
    for rname in round_names:
        round_name_progress[rname] = list(current_round)
        next_round = []
        for i in range(0, len(current_round), 2):
            winner = simulate_match_score(prob_table, current_round[i], current_round[i + 1])
            next_round.append(winner)
        current_round = next_round

    champion = current_round[0]
    return champion, round_name_progress


def main():
    model = joblib.load("data/match_model.pkl")
    elo = load_elo()

    all_teams = [t for teams in GROUPS.values() for t in teams]
    print(f"Berechne Wahrscheinlichkeiten für alle {len(all_teams)} Teams (einmalig)...")
    prob_table = precompute_all_probs(model, elo, all_teams)

    win_counts = defaultdict(int)
    stage_reached = defaultdict(lambda: defaultdict(int))

    print(f"Simuliere {N_SIMULATIONS:,} WM-2026-Turniere...")
    for sim in range(N_SIMULATIONS):
        champion, progress = run_single_tournament(prob_table)
        win_counts[champion] += 1
        for stage, teams in progress.items():
            for t in teams:
                stage_reached[t][stage] += 1

    results = []
    for team in win_counts.keys() | stage_reached.keys():
        results.append({
            "team": team,
            "win_pct": 100 * win_counts.get(team, 0) / N_SIMULATIONS,
            "reach_final_pct": 100 * stage_reached[team].get("F", 0) / N_SIMULATIONS,
            "reach_sf_pct": 100 * stage_reached[team].get("SF", 0) / N_SIMULATIONS,
            "reach_qf_pct": 100 * stage_reached[team].get("QF", 0) / N_SIMULATIONS,
        })

    result_df = pd.DataFrame(results).sort_values("win_pct", ascending=False).reset_index(drop=True)
    result_df.index += 1

    print("\n=== TOP 15 WM-2026-FAVORITEN (Modell-Vorhersage, Stand 10. Juni 2026) ===")
    print(result_df.head(15).to_string())

    result_df.to_csv("results/wm2026_predictions.csv", index=False)
    print("\n✅ Gespeichert: results/wm2026_predictions.csv")


if __name__ == "__main__":
    main()
