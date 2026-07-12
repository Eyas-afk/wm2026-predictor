"""
Erstellt ein Balkendiagramm der Top-10-WM-2026-Favoriten aus den
Simulationsergebnissen - z.B. für README oder LinkedIn-Post.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    pred = pd.read_csv(PROJECT_ROOT / "results" / "wm2026_predictions.csv")
    top10 = pred.sort_values("win_pct", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))

    # barh() = horizontales Balkendiagramm (Balken liegen, nicht stehen) -
    # dadurch sind lange Team-Namen besser lesbar als bei senkrechten Balken
    bars = ax.barh(top10["team"], top10["win_pct"], color="#1a5f2e")

    # Damit die stärksten Favoriten oben stehen, nicht unten (barh() sortiert
    # sonst von unten nach oben)
    ax.invert_yaxis()

    ax.set_xlabel("Titelwahrscheinlichkeit (%)")
    ax.set_title("WM 2026: Top 10 Favoriten laut Modell\n(Stand: 10. Juni 2026, vor Turnierstart)")

    # Prozentwert direkt neben jeden Balken schreiben, statt nur die
    # Achsenbeschriftung lesen zu müssen
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.15, bar.get_y() + bar.get_height() / 2,
                 f"{width:.1f}%", va="center")

    plt.tight_layout()
    output_path = PROJECT_ROOT / "results" / "top10_favorites.png"
    plt.savefig(output_path, dpi=150)
    print(f"✅ Gespeichert: {output_path}")


if __name__ == "__main__":
    main()