"""
Vergleicht die Modell-Vorhersage (trainiert nur mit Daten bis 10.06.2026)
mit dem tatsächlichen WM-2026-Turnierverlauf.

Das ist die "Feuerprobe" für unser Modell: Hätten wir mit dem Wissen von
vor dem Turnierstart richtig gelegen?
"""

import pandas as pd


def get_actual_round_of_16_teams(results_path="data/results.csv", shootouts_path="data/shootouts.csv"):
    df = pd.read_csv(results_path)
    so = pd.read_csv(shootouts_path)
    df["date"] = pd.to_datetime(df["date"])
    so["date"] = pd.to_datetime(so["date"])

    # Runde der letzten 32 lief vom 28.06. bis 03.07.2026 (16 Spiele)
    r32 = df[
        (df["tournament"] == "FIFA World Cup")
        & (df["date"] >= "2026-06-28")
        & (df["date"] <= "2026-07-03")
    ]

    def get_winner(row):
        if row["home_score"] > row["away_score"]:
            return row["home_team"]
        elif row["home_score"] < row["away_score"]:
            return row["away_team"]
        else:
            match = so[
                (so["date"] == row["date"])
                & (so["home_team"] == row["home_team"])
                & (so["away_team"] == row["away_team"])
            ]
            return match.iloc[0]["winner"] if len(match) else None

    winners = set(r32.apply(get_winner, axis=1))
    return winners


def main():
    pred = pd.read_csv("results/wm2026_predictions.csv")
    actual_r16 = get_actual_round_of_16_teams()

    top15 = set(pred.sort_values("win_pct", ascending=False).head(15)["team"])

    hits = sorted(top15 & actual_r16)
    overestimated = sorted(top15 - actual_r16)
    underestimated = sorted(actual_r16 - top15)

    print("=== VALIDIERUNG: Modell-Vorhersage vs. echter WM-2026-Verlauf ===\n")
    print(f"Tatsächliche Achtelfinal-Teams: {len(actual_r16)}")
    print(sorted(actual_r16))
    print()
    print(f"✅ Treffer ({len(hits)}/15 Top-Favoriten haben's ins Achtelfinale geschafft):")
    print(f"   {hits}")
    print()
    print(f"⚠️  Modell zu optimistisch (Top-15, real aber ausgeschieden):")
    print(f"   {overestimated}")
    print()
    print(f"📉 Modell hat unterschätzt (real im Achtelfinale, nicht in Top-15):")
    print(f"   {underestimated}")
    print()
    accuracy = len(hits) / 15
    print(f"Trefferquote Top-15 -> Achtelfinale: {accuracy:.1%}")


if __name__ == "__main__":
    main()
