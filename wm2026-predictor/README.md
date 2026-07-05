# WM 2026 Predictor 🏆

Ein Machine-Learning-Modell, das die Gewinnwahrscheinlichkeiten für die
Fußball-Weltmeisterschaft 2026 vorhersagt – basierend auf über 100 Jahren
historischer Länderspielergebnisse.

## Methode

1. **ELO-Rating**: Für jedes Nationalteam wird ein ELO-Rating aus der
   gesamten Spielhistorie berechnet (ähnlich wie im Schach), gewichtet nach
   Turnier-Wichtigkeit (WM > Kontinental-Cup > Quali > Freundschaftsspiel).
2. **Vorhersagemodell**: Ein Gradient-Boosting-Klassifikator sagt aus der
   ELO-Differenz + Kontext-Features (neutraler Ort etc.) die Wahrscheinlichkeit
   für Heimsieg / Unentschieden / Auswärtssieg voraus.
3. **Turniersimulation**: Das WM-2026-Format (48 Teams, 12 Gruppen) wird
   10.000x per Monte-Carlo-Simulation durchgespielt, um für jedes Team eine
   Gesamt-Titelwahrscheinlichkeit zu ermitteln.

## Daten

Quelle: [International football results (Kaggle)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
— 49.500+ internationale Spiele von 1872 bis heute.

## Setup

```bash
pip install -r requirements.txt
```

## Nutzung

```bash
python src/elo.py          # ELO-Ratings berechnen
python src/model.py        # Modell trainieren
python src/simulate.py     # WM 2026 simulieren (10.000 Monte-Carlo-Durchläufe)
python src/validate.py     # Vorhersage mit echtem Turnierverlauf vergleichen
```

## Besonderheit: Live-Validierung

Die WM 2026 läuft bereits (Start: 11. Juni 2026), als dieses Projekt entstand.
Statt das zu ignorieren, wurde daraus ein Feature: Das Modell wird
ausschließlich mit Daten bis zum 10. Juni 2026 (Tag vor Turnierstart)
trainiert – kein Data Leakage. Die 12 WM-Gruppen wurden aus den historischen
Spieldaten rekonstruiert, das komplette Turnier 10.000x simuliert, und die
Vorhersage anschließend mit dem tatsächlichen Turnierverlauf verglichen.

## Ergebnisse

Top 10 WM-2026-Favoriten laut Modell (Stand: 10. Juni 2026, vor Turnierstart):

| Rang | Team | Titelwahrscheinlichkeit |
|------|------|--------------------------|
| 1 | Spanien | 12.3% |
| 2 | Argentinien | 10.0% |
| 3 | Brasilien | 8.5% |
| 4 | Frankreich | 7.9% |
| 5 | Kolumbien | 7.7% |
| 6 | England | 6.8% |
| 7 | Deutschland | 4.7% |
| 8 | Niederlande | 4.6% |
| 9 | Portugal | 4.4% |
| 10 | Japan | 4.2% |

### Validierung gegen die Realität

Vergleich: Modell-Top-15 vs. tatsächliche Achtelfinal-Teilnehmer (echte
Ergebnisse bis 4. Juli 2026, inkl. Elfmeterschießen):

- **Treffer**: 10 von 15 Top-Favoriten haben tatsächlich das Achtelfinale
  erreicht (Spanien, Argentinien, Brasilien, Frankreich, Kolumbien, England,
  Portugal, Marokko, Norwegen, Schweiz) → **66.7% Trefferquote**
- **Überschätzt**: Deutschland, Niederlande, Japan, Ecuador, Kroatien (alle
  ausgeschieden, teils im Elfmeterschießen)
- **Unterschätzt**: Belgien, Kanada, Ägypten, Mexiko, Paraguay, USA schafften
  es ins Achtelfinale, obwohl das Modell sie nicht in den Top 15 hatte

**Fazit**: Ein reines ELO-basiertes Modell erfasst die grundsätzliche
Team-Stärke gut, aber K.o.-Runden im Fußball bleiben durch Elfmeterschießen
und Tagesform inhärent volatil – das bestätigt sich hier eindrucksvoll.

## Modell-Performance

- Accuracy (Sieg/Remis/Niederlage): 57.6% (Baseline "immer Heimsieg": 49.0%)
- Wichtigstes Feature: ELO-Differenz (84.7% Feature Importance)
- Schwäche: Unentschieden sind praktisch nicht vorhersagbar (bekanntes
  Problem im Fußball-Modeling)

## Projektstruktur

```
wm2026-predictor/
├── data/               # Rohdaten + generierte ELO-Ratings/Modell
├── src/
│   ├── elo.py          # ELO-Rating-Berechnung
│   ├── model.py        # Modelltraining
│   ├── simulate.py     # Monte-Carlo-Turniersimulation
│   └── validate.py     # Vergleich mit echtem Turnierverlauf
└── results/
    └── wm2026_predictions.csv
```

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (Gradient Boosting)
- Monte-Carlo-Simulation
- networkx (Gruppen-Rekonstruktion aus Spieldaten)

## Mögliche Erweiterungen

- Echte K.o.-Bracket-Struktur statt vereinfachter zufälliger Paarung
- Zusätzliche Features: Kaderwert, aktuelle Form, Verletzungen
- Poisson-Regression für exakte Ergebnis-Vorhersagen (nicht nur 1X2)
- Live-Update während des laufenden Turniers
