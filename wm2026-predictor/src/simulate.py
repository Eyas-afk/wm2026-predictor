"""
Simuliert die WM 2026 (48 Teams, 12 Gruppen à 4) per Monte-Carlo-Verfahren,
basierend auf dem trainierten Modell und den ELO-Ratings vom 10. Juni 2026
(letzter Tag vor Turnierstart -> kein Data Leakage).

Format: 12 Gruppen -> Gruppensieger + Gruppenzweite (24) + 8 beste Dritte
        -> 32 Teams -> K.o.-Runde (Runde der letzten 32 -> Achtelfinale ->
        Viertelfinale -> Halbfinale -> Finale)
"""

import pandas as pd
import numpy as np
import joblib
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Anzahl kompletter Turnier-Durchläufe. Je höher, desto stabiler/genauer die
# geschätzten Wahrscheinlichkeiten (Monte-Carlo-Prinzip: viele Wiederholungen
# nähern sich der "wahren" Wahrscheinlichkeit an) - aber auch länger Laufzeit.
# 10.000 ist ein guter Kompromiss zwischen Genauigkeit und Rechenzeit.
N_SIMULATIONS = 10_000
# Fester Seed (42) macht die Simulation reproduzierbar - gleicher Lauf,
# gleiches Ergebnis. Wichtig, um Modelländerungen später fair vergleichen
# zu können, ohne dass Zufallsschwankungen die Interpretation verfälschen.
RNG = np.random.default_rng(42)

# Die 12 Gruppen, wie im echten WM-2026-Turnier. Aus den historischen
# Spieldaten rekonstruiert (siehe Gruppenphasen-Spiele Juni 2026), nicht
# manuell recherchiert - stellt sicher, dass die Gruppen exakt zu den
# ELO-Werten/Daten passen, mit denen wir trainiert haben.
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
    """Lädt die finalen ELO-Ratings (Stand 10. Juni 2026) aus elo.py als
    einfaches {team_name: rating}-Dictionary für schnellen Zugriff."""
    elo = pd.read_csv(PROJECT_ROOT / "data" / "final_elo_ratings.csv", index_col=0).iloc[:, 0].to_dict()
    return elo


def precompute_all_probs(model, elo, teams):
    """
    Berechnet EINMAL alle paarweisen Wahrscheinlichkeiten für alle Teams im
    Turnier (statt bei jeder Simulation neu). Das ist der Performance-Trick,
    der 10.000 Simulationen erst praktikabel macht.

    Ohne diesen Trick: ~103 Spiele x 10.000 Simulationen = über 1 Million
    einzelne model.predict_proba()-Aufrufe -> viel zu langsam.
    Mit diesem Trick: 1 einziger Aufruf für alle 48x47 = 2.256 möglichen
    Paarungen auf einmal, danach nur noch Dictionary-Lookups (quasi
    augenblicklich). Funktioniert, weil die ELO-Werte während der gesamten
    Simulation fix bleiben - die Wahrscheinlichkeit "Spanien vs. Brasilien"
    ändert sich nicht zwischen den 10.000 Durchläufen.
    """
    rows = []
    pairs = []
    for a in teams:
        for b in teams:
            if a == b:
                continue
            ra, rb = elo.get(a, 1500), elo.get(b, 1500)
            # neutral=1 fest gesetzt: bei der WM (außer für Gastgeberländer)
            # gibt es praktisch keinen echten Heimvorteil - jedes Spiel findet
            # auf "fremdem" Boden für beide Teams statt.
            rows.append({"elo_diff": ra - rb, "home_elo_pre": ra, "away_elo_pre": rb, "neutral": 1})
            pairs.append((a, b))

    X = pd.DataFrame(rows)
    probs = model.predict_proba(X)  # (n, 3) -> [p_home, p_draw, p_away]

    # Dictionary mit Tupel-Keys (team_a, team_b) für O(1)-Lookup während der
    # Simulation. Achtung: (a, b) und (b, a) sind unterschiedliche Einträge,
    # weil elo_diff sich je nach Reihenfolge im Vorzeichen umdreht.
    prob_table = {}
    for (a, b), (p_a, p_d, p_b) in zip(pairs, probs):
        prob_table[(a, b)] = (p_a, p_d, p_b)
    return prob_table


def simulate_match_score(prob_table, team_a, team_b):
    """
    Simuliert ein einzelnes K.o.-Spiel. Anders als in der Gruppenphase kann
    ein K.o.-Spiel nicht unentschieden enden (nach Verlängerung würde ein
    Elfmeterschießen entscheiden) - deshalb wird die Unentschieden-
    Wahrscheinlichkeit hier hälftig auf beide Teams verteilt (vereinfachte
    Annahme: Elfmeterschießen ist ungefähr 50/50, unabhängig von der
    Team-Stärke).
    """
    p_a, p_d, p_b = prob_table[(team_a, team_b)]
    p_a_wins_knockout = p_a + p_d / 2
    return team_a if RNG.random() < p_a_wins_knockout else team_b


def simulate_group(prob_table, teams):
    """
    Simuliert alle 6 Spiele einer 4er-Gruppe (jedes Team spielt einmal gegen
    jedes andere) und gibt die finale Tabelle zurück.

    Rückgabe:
        ranking: Teams sortiert von Platz 1 bis 4
        points, goal_diff: Rohdaten für die Sortierung, werden in
                            run_single_tournament() für die "beste 8
                            Gruppendritte"-Regel weiterverwendet
    """
    points = {t: 0 for t in teams}
    # Vereinfachung: Tordifferenz wird nur als +1/-1 pro Spielausgang
    # gezählt (nicht die tatsächliche Anzahl Tore, da unser Modell nur
    # Sieg/Remis/Niederlage vorhersagt, keine exakten Spielstände). Dient
    # hier nur als Tie-Break zwischen punktgleichen Teams, keine exakte
    # Nachbildung echter Tordifferenzen.
    goal_diff = {t: 0 for t in teams}

    # i, j-Schleife mit j startet bei i+1: erzeugt jedes Team-Paar genau
    # einmal (kein Team gegen sich selbst, keine doppelten Paarungen wie
    # sowohl (A,B) als auch (B,A)). Bei 4 Teams ergibt das exakt 6 Spiele.
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            p_a, p_d, p_b = prob_table[(a, b)]
            # Der eigentliche Zufalls-"Würfelwurf" dieses Spiels, gewichtet
            # nach den Modell-Wahrscheinlichkeiten.
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

    # Sortierung nach Tupel (points, goal_diff): Punkte haben Vorrang,
    # Tordifferenz entscheidet nur bei Punktgleichstand (Tie-Break) -
    # genau wie in echten Fußball-Tabellen. reverse=True für absteigend
    # (Erster oben).
    ranking = sorted(teams, key=lambda t: (points[t], goal_diff[t]), reverse=True)
    return ranking, points, goal_diff


def run_single_tournament(prob_table):
    """
    Simuliert EIN komplettes WM-Turnier (alle 12 Gruppen + komplette
    K.o.-Runde) und gibt den Champion zurück, sowie welche Teams welche
    Runde erreicht haben. Wird in main() 10.000x aufgerufen.
    """
    group_results = {}
    all_third_placed = []

    for g_name, teams in GROUPS.items():
        ranking, points, gd = simulate_group(prob_table, teams)
        group_results[g_name] = ranking
        # ranking[2] = Gruppendritter (Index 0=Erster, 1=Zweiter, 2=Dritter).
        # Wird separat gesammelt, weil beim WM-Format die 8 besten von 12
        # Gruppendritten ebenfalls weiterkommen.
        all_third_placed.append((ranking[2], points[ranking[2]], gd[ranking[2]]))

    # 8 beste Gruppendritte kommen weiter (echte WM-2026-Regel bei 12
    # Gruppen). Sortierung nach demselben (Punkte, Tordifferenz)-Prinzip
    # wie innerhalb einer Gruppe.
    best_thirds = sorted(all_third_placed, key=lambda x: (x[1], x[2]), reverse=True)[:8]
    best_third_teams = [t[0] for t in best_thirds]

    # 32 Teams für die K.o.-Runde: 12 Gruppen x (Erster + Zweiter) = 24,
    # plus 8 beste Dritte = 32.
    round_of_32 = []
    for ranking in group_results.values():
        round_of_32.extend([ranking[0], ranking[1]])
    round_of_32.extend(best_third_teams)

    # WICHTIGE VEREINFACHUNG: Ein echtes WM-Bracket hat feste Paarungsregeln
    # (z.B. Gruppe-A-Erster gegen Gruppe-B-Zweiter), damit z.B. zwei
    # Top-Teams aus derselben Gruppe nicht schon im Achtelfinale aufeinander-
    # treffen. Hier wird stattdessen zufällig gemischt - technisch nicht
    # 1:1 wie das echte Turnier, aber für den Zweck dieses Lernprojekts
    # (Grundprinzip von Monte-Carlo-Turniersimulation zeigen) ausreichend.
    # Mögliche Erweiterung: echte Bracket-Logik nachbauen.
    bracket = list(round_of_32)
    RNG.shuffle(bracket)

    # Protokolliert, welche Teams in welcher Runde noch dabei waren -
    # wird später für "reach_final_pct" etc. in main() ausgewertet.
    round_name_progress = defaultdict(list)
    current_round = bracket
    round_names = ["R32", "R16", "QF", "SF", "F"]
    for rname in round_names:
        round_name_progress[rname] = list(current_round)
        next_round = []
        # Paarweise durch die aktuelle Runde: Team 0 vs Team 1, Team 2 vs
        # Team 3, usw. Jedes Paar erzeugt genau einen Sieger für die
        # nächste Runde.
        for i in range(0, len(current_round), 2):
            winner = simulate_match_score(prob_table, current_round[i], current_round[i + 1])
            next_round.append(winner)
        current_round = next_round

    # Nach der letzten Runde ("F" = Finale) bleibt nur noch 1 Team übrig.
    champion = current_round[0]
    return champion, round_name_progress


def main():
    model = joblib.load(PROJECT_ROOT / "data" / "match_model.pkl")
    elo = load_elo()

    all_teams = [t for teams in GROUPS.values() for t in teams]
    print(f"Berechne Wahrscheinlichkeiten für alle {len(all_teams)} Teams (einmalig)...")
    # Performance-kritischer Schritt: einmal berechnen, danach nur noch
    # nachschlagen (siehe Docstring von precompute_all_probs).
    prob_table = precompute_all_probs(model, elo, all_teams)

    win_counts = defaultdict(int)
    # Verschachteltes defaultdict: stage_reached["Spain"]["QF"] zählt, in
    # wie vielen der 10.000 Simulationen Spanien das Viertelfinale erreicht hat.
    stage_reached = defaultdict(lambda: defaultdict(int))

    print(f"Simuliere {N_SIMULATIONS:,} WM-2026-Turniere...")
    for sim in range(N_SIMULATIONS):
        champion, progress = run_single_tournament(prob_table)
        win_counts[champion] += 1
        for stage, teams in progress.items():
            for t in teams:
                stage_reached[t][stage] += 1

    # Ergebnisse in eine übersichtliche Tabelle umwandeln: für jedes Team,
    # das irgendwann in mind. einer Simulation vorkam, berechnen wir den
    # prozentualen Anteil der Simulationen, in denen es Champion wurde bzw.
    # eine bestimmte Runde erreicht hat.
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
    result_df.index += 1  # Rang 1 statt Index 0, für lesbarere Ausgabe

    print("\n=== TOP 15 WM-2026-FAVORITEN (Modell-Vorhersage, Stand 10. Juni 2026) ===")
    print(result_df.head(15).to_string())

    # Für validate.py und die spätere README-Tabelle persistieren.
    result_df.to_csv(PROJECT_ROOT / "results" / "wm2026_predictions.csv", index=False)
    print("\n✅ Gespeichert: results/wm2026_predictions.csv")


if __name__ == "__main__":
    main()